package com.modelforge.app.nativebridge

import com.modelforge.app.core.ModelConfig

/**
 * Read-only quantized model loaded from a packed export (MFPK).
 * Linear layers stay packed in memory; embeddings, norms, biases and the tied
 * output head stay fp32. All calls are serialized.
 */
class PackedModel private constructor(
    private var handle: Long,
    val config: ModelConfig,
    val bits: Int,
    val group: Int
) : AutoCloseable {

    val paramCount: Long
        @Synchronized get() = if (handle == 0L) 0L else NativeBridge.nativePackedParamCount(handle)

    val isOpen: Boolean get() = handle != 0L

    @Synchronized
    fun generate(
        prompt: IntArray,
        maxNew: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: Long = System.nanoTime()
    ): IntArray {
        check(handle != 0L) { "Packed model is closed" }
        return NativeBridge.nativePackedGenerate(handle, prompt, maxNew, temperature, topK, topP, seed)
            ?: IntArray(0)
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            NativeBridge.nativePackedFree(handle)
            handle = 0L
        }
    }

    companion object {
        fun load(path: String): PackedModel? {
            val handle = NativeBridge.nativeLoadPacked(path)
            if (handle == 0L) return null
            val values = NativeBridge.nativePackedConfig(handle) ?: run {
                NativeBridge.nativePackedFree(handle)
                return null
            }
            val config = ModelConfig(
                vocabSize = values[0],
                contextLength = values[1],
                dModel = values[2],
                layers = values[3],
                heads = values[4],
                dFf = values[5],
                tiedEmbeddings = values[6] != 0,
                bits = values[7],
                qat = values[7] in 2..8
            )
            return PackedModel(handle, config, values[7], values[8])
        }
    }
}
