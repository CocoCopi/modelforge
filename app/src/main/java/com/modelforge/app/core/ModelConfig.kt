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
    val qat: Boolean = false
) {
    val headDim: Int get() = if (heads > 0) dModel / heads else 0
    val paramsPerHead: Int get() = headDim

    fun isValid(): Boolean =
        vocabSize > 0 && contextLength > 0 && dModel > 0 && layers > 0 &&
            heads > 0 && dModel % heads == 0 && dFf > 0

    /** Exact parameter count. Must stay in sync with Model::paramCount() in C++. */
    fun paramCount(): Long {
        if (!isValid()) return 0L
        val d = dModel.toLong()
        val df = dFf.toLong()
        val v = vocabSize.toLong()
        val c = contextLength.toLong()
        val l = layers.toLong()
        val perLayer = 4L * d * d + 2L * d * df + 9L * d + df
        val head = if (tiedEmbeddings) 0L else v * d
        return v * d + c * d + l * perLayer + 2L * d + head
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
        val LAYER_BITS = intArrayOf(2, 3, 4, 8, 16, 32)

        val BIT_EXPLANATIONS = mapOf(
            32 to "Full fp32. Largest files, best quality, no quantization-aware training.",
            16 to "Half-precision linear weights. Tiny quality loss, ~2x smaller than fp32.",
            8 to "8-bit linear weights. Very good quality, ~4x smaller. QAT keeps it stable.",
            4 to "4-bit linear weights. The usual sweet spot on phones, ~8x smaller.",
            3 to "3-bit linear weights. Very small, noticeably rougher text.",
            2 to "2-bit linear weights. Experimental and smallest, expect degraded text."
        )
    }
}
