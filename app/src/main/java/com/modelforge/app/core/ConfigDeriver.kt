package com.modelforge.app.core

import kotlin.math.abs

/**
 * Finds a realistic GPT architecture whose *counted* parameter count lands within
 * 5% of a target. Pure Kotlin so it is unit-testable without Android.
 */
object ConfigDeriver {

    data class Options(
        val maxParams: Long = 200_000_000L,
        val contextCandidates: IntArray = intArrayOf(128, 256, 512),
        val vocabCandidates: IntArray = intArrayOf(1024, 2048, 4096, 8192, 16384),
        val headDims: IntArray = intArrayOf(16, 32, 64),
        val maxLayers: Int = 24
    )

    data class Result(
        val config: ModelConfig,
        val errorFraction: Double,
        val withinTolerance: Boolean,
        val explanation: String
    )

    fun derive(targetParams: Long, options: Options = Options()): Result {
        require(targetParams > 0) { "targetParams must be positive" }
        val cappedTarget = targetParams.coerceAtMost(options.maxParams)
        require(cappedTarget > 0) { "maxParams must be positive" }

        var bestScore = Double.MAX_VALUE
        var best: Candidate? = null
        var closestErr = Double.MAX_VALUE
        var closest: Candidate? = null

        for (context in options.contextCandidates) {
            for (vocab in options.vocabCandidates) {
                for (headDim in options.headDims) {
                    for (heads in 1..16) {
                        val d = headDim * heads
                        if (d > 2048) continue
                        val dFf = 4 * d
                        for (layers in 1..options.maxLayers) {
                            val cfg = ModelConfig(
                                vocabSize = vocab,
                                contextLength = context,
                                dModel = d,
                                layers = layers,
                                heads = heads,
                                dFf = dFf,
                                targetParams = cappedTarget
                            )
                            val count = cfg.paramCount()
                            if (count > options.maxParams) continue
                            val err = abs(count - cappedTarget).toDouble() / cappedTarget.toDouble()
                            if (err < closestErr) {
                                closestErr = err
                                closest = Candidate(cfg, count, err)
                            }
                            if (err > 0.045) continue
                            val score = aestheticPenalty(vocab, context, d, layers, heads, headDim) + err * 0.5
                            if (score < bestScore) {
                                bestScore = score
                                best = Candidate(cfg, count, err)
                            }
                        }
                    }
                }
            }
        }

        val chosen = best ?: closest
            ?: throw IllegalStateException("No architecture fits within ${options.maxParams} parameters")
        val within = chosen.err <= 0.05
        val explanation = buildString {
            append("d_model=${chosen.cfg.dModel}, ${chosen.cfg.layers} layers, ")
            append("${chosen.cfg.heads} heads (${chosen.cfg.headDim}/head), ")
            append("d_ff=${chosen.cfg.dFf}, context=${chosen.cfg.contextLength}, ")
            append("vocab=${chosen.cfg.vocabSize}")
            if (cappedTarget < targetParams) {
                append(". Target capped by device limit to ${formatCount(cappedTarget)} parameters.")
            }
        }
        return Result(chosen.cfg, chosen.err, within, explanation)
    }

    private fun aestheticPenalty(
        vocab: Int,
        context: Int,
        d: Int,
        layers: Int,
        heads: Int,
        headDim: Int
    ): Double {
        var p = 0.0
        p += when (headDim) {
            64 -> 0.0
            32 -> 0.004
            else -> 0.012
        }
        if (heads < 4) p += 0.02
        else if (heads and (heads - 1) != 0) p += 0.003
        when {
            layers < 4 -> p += 0.012
            layers > 16 -> p += 0.004
        }
        when (vocab) {
            1024 -> p += 0.003
            2048, 4096 -> Unit
            8192 -> p += 0.003
            else -> p += 0.006
        }
        if (context != 256) p += 0.003
        if (d % 64 != 0) p += 0.002
        return p
    }

    private data class Candidate(val cfg: ModelConfig, val count: Long, val err: Double)

    fun formatCount(n: Long): String = when {
        n >= 1_000_000_000L -> String.format("%.2fB", n / 1_000_000_000.0)
        n >= 1_000_000L -> String.format("%.2fM", n / 1_000_000.0)
        n >= 1_000L -> String.format("%.1fK", n / 1_000.0)
        else -> n.toString()
    }

    fun formatBytes(bytes: Long): String = when {
        bytes >= 1L shl 30 -> String.format("%.2f GB", bytes / (1L shl 30).toDouble())
        bytes >= 1L shl 20 -> String.format("%.1f MB", bytes / (1L shl 20).toDouble())
        bytes >= 1L shl 10 -> String.format("%.1f KB", bytes / (1L shl 10).toDouble())
        else -> "$bytes B"
    }
}
