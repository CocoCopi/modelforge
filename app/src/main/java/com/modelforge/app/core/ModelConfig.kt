package com.modelforge.app.core

import kotlinx.serialization.Serializable

/**
 * Architecture of a small GPT model. This is the single source of truth shared by
 * Kotlin (UI, config files) and the C++ engine (which uses the same parameter formula).
 */
@Serializable
data class ModelConfig(
    val vocabSize: Int,
    val contextLength: Int,
    val dModel: Int,
    val layers: Int,
    val heads: Int,
    val dFf: Int,
    val tiedEmbeddings: Boolean = true,
    val targetParams: Long = 0L,
    val bits: Int = 32,
    val qat: Boolean = false,
    val architectureId: String = ArchitectureRegistry.GPT_TEXT
) {
    val headDim: Int get() = if (heads > 0) dModel / heads else 0
    val paramsPerHead: Int get() = headDim

    // The engine has no separate output head: logits come from the tied
    // embedding matrix. This mirrors ModelConfig::valid() in C++ so a config
    // that would be rejected natively can never be written to a manifest.
    fun isValid(): Boolean =
        tiedEmbeddings && vocabSize > 0 && contextLength > 0 && dModel > 0 &&
            layers > 0 && heads > 0 && dModel % heads == 0 && dFf > 0

    /** Exact parameter count. Must stay in sync with Model::paramCount() in C++. */
    fun paramCount(): Long {
        if (!isValid()) return 0L
        val d = dModel.toLong()
        val df = dFf.toLong()
        val v = vocabSize.toLong()
        val l = layers.toLong()
        val perLayer = 4L * d * d + 2L * d * df + 9L * d + df
        // Positions are fixed sinusoidal values, so context length adds no
        // trainable parameters and no context-sized position table.
        return v * d + l * perLayer + 2L * d
    }

    fun linearWeightCount(): Long {
        if (!isValid()) return 0L
        val d = dModel.toLong()
        val df = dFf.toLong()
        return layers.toLong() * (4L * d * d + 2L * d * df)
    }

    fun parameterErrorFraction(actual: Long = paramCount()): Double {
        if (targetParams <= 0L) return 0.0
        return kotlin.math.abs(actual - targetParams).toDouble() / targetParams.toDouble()
    }

    /** Bytes needed for training (fp32 master weights + grads + AdamW m/v + overhead). */
    fun trainingBytes(): Long = paramCount() * 20L + 64L * 1024L * 1024L

    /** Bytes saved by a 32-bit float checkpoint (weights only). */
    fun fp32Bytes(): Long = paramCount() * 4L

    /**
     * Size of the packed export, matching the C++ exporter layout.
     * Linear weights use packed values + one fp32 scale per group of [groupSize]
     * values; embeddings, norms, biases and the tied head stay fp32.
     */
    fun packedExportBytes(bits: Int, groupSize: Int = 32): Long {
        if (!isValid()) return 0L
        if (bits !in LAYER_BITS) return fp32Bytes()
        val d = dModel.toLong()
        val df = dFf.toLong()
        val l = layers.toLong()
        val gs = groupSize.toLong()

        val linearWeights = linearWeightCount()
        val nonLinear = paramCount() - linearWeights

        // per layer: wq, wk, wv, wo [d,d], w1 [df,d], w2 [d,df]
        val perLayerScales = 4L * d * ((d + gs - 1) / gs) +
            df * ((d + gs - 1) / gs) +
            d * ((df + gs - 1) / gs)
        val scales = l * perLayerScales

        val packedData = (linearWeights * bits + 7L) / 8L
        return nonLinear * 4L + scales * 4L + packedData
    }

    fun packedSavingsVsFp32(bits: Int): Double {
        val fp = fp32Bytes().toDouble()
        if (fp <= 0.0) return 0.0
        return 1.0 - packedExportBytes(bits).toDouble() / fp
    }

    companion object {
        /**
         * Widths the packed (MFPK) exporter and QAT path actually implement.
         * F16/F32 are export choices in the GGUF and safetensors menus, not
         * training-compression choices, so they are deliberately absent here.
         */
        val LAYER_BITS = intArrayOf(2, 3, 4, 8)

        val BIT_EXPLANATIONS = mapOf(
            8 to "8-bit linear weights. Very good quality, ~4x smaller. QAT keeps it stable.",
            4 to "4-bit linear weights. The usual sweet spot on phones, ~8x smaller.",
            3 to "3-bit linear weights. Very small, noticeably rougher text.",
            2 to "2-bit linear weights. Experimental and smallest, expect degraded text."
        )
    }
}
