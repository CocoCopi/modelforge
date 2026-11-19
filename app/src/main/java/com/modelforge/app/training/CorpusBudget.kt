package com.modelforge.app.training

import kotlin.math.max
import kotlin.math.min

/**
 * How much text the trainer loads into memory for one run.
 *
 * The old hard clamp at 4,000,000 characters silently threw away most of a
 * large dataset: a 132k-row corpus would be cut to a few thousand rows, so
 * every step after the first epoch repeated the same small slice. The budget
 * here is derived from the requested number of steps and from RAM, not from an
 * arbitrary constant, so long runs actually see the data the user downloaded.
 */
object CorpusBudget {
    const val MIN_CHARS = 200_000L

    /** Absolute ceiling: a 128M-char corpus is ~256 MB of String plus tokens. */
    const val MAX_CHARS = 128_000_000L

    /**
     * Largest corpus that leaves room for weights, AdamW moments, activations
     * and the tokenizer. A Kotlin String costs 2 bytes per char and the token
     * array roughly 1 byte per char for ordinary text.
     */
    fun maxCharsFor(availableRamBytes: Long): Long {
        if (availableRamBytes <= 0L) return 8_000_000L
        val usable = availableRamBytes / 6L
        return (usable / 3L).coerceIn(MIN_CHARS, MAX_CHARS)
    }

    /**
     * Characters needed for roughly one pass without immediate repetition:
     * about three characters per trained token.
     */
    fun forSteps(steps: Int, batchSize: Int, seqLength: Int, maxChars: Long = MAX_CHARS): Int {
        val safeSteps = max(1, steps).toLong()
        val safeBatch = max(1, batchSize).toLong()
        val safeSeq = max(1, seqLength).toLong()
        val needed = safeSteps * safeBatch * safeSeq * 3L
        val ceiling = max(MIN_CHARS, min(maxChars, MAX_CHARS))
        return needed.coerceIn(MIN_CHARS, ceiling).toInt()
    }

    /**
     * Final corpus size for a run: never below what one pass over `steps`
     * needs and never above what RAM can hold. A user- or AI-supplied budget
     * is honoured as a minimum, so requesting more text than the step count
     * strictly needs is still allowed.
     */
    fun resolve(
        requested: Int,
        steps: Int,
        batchSize: Int,
        seqLength: Int,
        availableRamBytes: Long
    ): Int {
        val ceiling = maxCharsFor(availableRamBytes)
        val needed = forSteps(steps, batchSize, seqLength, ceiling).toLong()
        val wanted = max(0, requested).toLong()
        return max(needed, wanted).coerceIn(MIN_CHARS, ceiling).toInt()
    }
}
