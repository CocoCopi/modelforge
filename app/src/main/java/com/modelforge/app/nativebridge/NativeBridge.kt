package com.modelforge.app.nativebridge

/**
 * Raw JNI surface. Handles are native pointers; always free them.
 * Higher-level code should use NativeEngine instead of calling this directly.
 */
object NativeBridge {
    init {
        System.loadLibrary("modelforge")
    }

    external fun nativeVersion(): String

    external fun nativeCreate(
        vocab: Int,
        context: Int,
        dModel: Int,
        layers: Int,
        heads: Int,
        dFf: Int,
        tied: Boolean,
        seed: Long
    ): Long

    external fun nativeFree(handle: Long)

    external fun nativeParamCount(handle: Long): Long

    external fun nativeConfig(handle: Long): IntArray?

    external fun nativeTrainStep(
        handle: Long,
        tokens: IntArray,
        targets: IntArray,
        batch: Int,
        seq: Int,
        learningRate: Float,
        weightDecay: Float
    ): Float

    external fun nativeStep(handle: Long): Long

    external fun nativeSetQAT(handle: Long, bits: Int, group: Int)

    external fun nativeGenerate(
        handle: Long,
        prompt: IntArray,
        maxNew: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        seed: Long
    ): IntArray?

    external fun nativeSaveCheckpoint(handle: Long, path: String, withOptimizer: Boolean): Boolean

    external fun nativeLoadCheckpoint(path: String, withOptimizer: Boolean): Long

    fun version(): String = nativeVersion()
}
