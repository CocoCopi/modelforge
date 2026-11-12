package com.modelforge.app.core

data class ReasoningText(
    val thinking: String?,
    val answer: String
)

object ResponseModes {
    const val INSTANT = "instant"
    const val NORMAL = "normal"
    const val EXTRA = "extra"

    const val INSTANT_MAX_TOKENS = 32
    const val EXTRA_MAX_TOKENS = 512

    fun normalize(mode: String?): String = when (mode) {
        INSTANT, NORMAL, EXTRA -> mode
        else -> NORMAL
    }

    fun label(mode: String): String = when (normalize(mode)) {
        INSTANT -> "Instant"
        EXTRA -> "Extra time"
        else -> "Normal"
    }

    /**
     * Response speed changes the inference budget, not the weights. Instant is a
     * short reply; extra time lets the model finish longer answers.
     */
    fun maxNewTokens(requested: Int, mode: String): Int {
        val base = requested.coerceIn(1, 512)
        return when (normalize(mode)) {
            INSTANT -> base.coerceAtMost(INSTANT_MAX_TOKENS)
            EXTRA -> base.coerceAtLeast(EXTRA_MAX_TOKENS)
            else -> base
        }
    }

    fun shouldThink(reasoningEnabled: Boolean, mode: String): Boolean =
        reasoningEnabled && normalize(mode) != INSTANT

    fun prompt(prompt: String, reasoningEnabled: Boolean, mode: String): String {
        if (!shouldThink(reasoningEnabled, mode)) return prompt
        return prompt.trimEnd() +
            "\n\nThink through the answer first inside  thinking...</think>, then write the final answer."
    }

    fun splitThinking(text: String): ReasoningText {
        val open = "<think>"
        val start = text.indexOf(open)
        if (start < 0) return ReasoningText(null, text)
        val contentStart = start + open.length
        val end = text.indexOf("</think>", contentStart)
        if (end < 0) {
            return ReasoningText(text.substring(contentStart).trimStart(), text.substring(0, start))
        }
        val thinking = text.substring(contentStart, end).trim()
        val answer = (text.substring(0, start) + text.substring(end + 8)).trim()
        return ReasoningText(thinking, answer)
    }
}
