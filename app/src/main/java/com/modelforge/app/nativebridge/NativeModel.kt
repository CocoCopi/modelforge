package com.modelforge.app.nativebridge

import com.modelforge.app.core.ModelConfig
import java.util.concurrent.locks.ReentrantReadWriteLock

/**
 * Thread-safe Kotlin wrapper around one native GPT model.
 * All operations are serialized because a training step mutates the model.
 *
 * [requestStop] is the one exception: it must reach the engine while a step is
 * running, so it deliberately does not take the per-model monitor. A read/write
 * lock only guards the native handle against being freed concurrently.
 */
class NativeModel private constructor(
    initialHandle: Long,
    var config: ModelConfig,
    var quantBits: Int,
    var quantGroup: Int
) : AutoCloseable {

    /** Guards [handle] against [close] racing [requestStop]. */
    private val lifecycle = ReentrantReadWriteLock()

    @Volatile
    private var handle: Long = initialHandle

    val paramCount: Long
        @Synchronized get() = if (handle == 0L) 0L else NativeBridge.nativeParamCount(handle)

    val step: Long
        @Synchronized get() = if (handle == 0L) 0L else NativeBridge.nativeStep(handle)

    val isOpen: Boolean get() = handle != 0L

    @Synchronized
    fun trainStep(tokens: IntArray, targets: IntArray, batch: Int, seq: Int, lr: Float, weightDecay: Float): Float {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeTrainStep(handle, tokens, targets, batch, seq, lr, weightDecay)
    }

    /**
     * Cooperatively abandons the current native step at the next layer
     * boundary. Returns immediately; the training loop observes
     * [STOPPED] from [trainStep] once the native side has unwound.
     */
    fun requestStop() {
        if (!lifecycle.readLock().tryLock()) return
        try {
            val h = handle
            if (h != 0L) NativeBridge.nativeRequestStop(h)
        } finally {
            lifecycle.readLock().unlock()
        }
    }

    /** Clears a previously requested stop so the model can be reused. */
    @Synchronized
    fun clearStop() {
        if (handle != 0L) NativeBridge.nativeClearStop(handle)
    }

    @Synchronized
    fun generate(
        prompt: IntArray,
        maxNew: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: Long = System.nanoTime()
    ): IntArray {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeGenerate(handle, prompt, maxNew, temperature, topK, topP, seed)
            ?: IntArray(0)
    }

    @Synchronized
    fun setQuantization(bits: Int, group: Int = 32) {
        check(handle != 0L) { "Model is closed" }
        NativeBridge.nativeSetQAT(handle, bits, group)
        quantBits = bits
        quantGroup = group
        config = config.copy(bits = bits, qat = bits in 2..8)
    }

    @Synchronized
    fun saveCheckpoint(path: String, withOptimizer: Boolean): Boolean {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeSaveCheckpoint(handle, path, withOptimizer)
    }

    @Synchronized
    fun savePacked(path: String, bits: Int, group: Int = 32): Boolean {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeSavePacked(handle, path, bits, group)
    }

    /** GGML tensor type: 0 = F32, 1 = F16, 2 = Q4_0, 8 = Q8_0. */
    @Synchronized
    fun saveGguf(path: String, ggmlType: Int): Boolean {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeSaveGguf(handle, path, ggmlType)
    }

    @Synchronized
    fun saveSafetensors(path: String, fp16: Boolean): Boolean {
        check(handle != 0L) { "Model is closed" }
        return NativeBridge.nativeSaveSafetensors(handle, path, fp16)
    }

    @Synchronized
    override fun close() {
        lifecycle.writeLock().lock()
        try {
            if (handle != 0L) {
                val h = handle
                handle = 0L
                NativeBridge.nativeFree(h)
            }
        } finally {
            lifecycle.writeLock().unlock()
        }
    }

    companion object {
        /** Returned by [trainStep] when [requestStop] aborted the step. */
        const val STOPPED = -2.0f

        fun create(config: ModelConfig, seed: Long = System.nanoTime()): NativeModel? {
            val handle = NativeBridge.nativeCreate(
                config.vocabSize,
                config.contextLength,
                config.dModel,
                config.layers,
                config.heads,
                config.dFf,
                config.tiedEmbeddings,
                seed
            )
            if (handle == 0L) return null
            return NativeModel(handle, config, config.bits, 32)
        }

        fun load(path: String, withOptimizer: Boolean = false): NativeModel? {
            val handle = NativeBridge.nativeLoadCheckpoint(path, withOptimizer)
            if (handle == 0L) return null
            val values = NativeBridge.nativeConfig(handle) ?: run {
                NativeBridge.nativeFree(handle)
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
                bits = values[7].let { if (it == 0) 32 else it },
                qat = values[7] in 2..8
            )
            return NativeModel(handle, config, values[7], values[8])
        }
    }
}
