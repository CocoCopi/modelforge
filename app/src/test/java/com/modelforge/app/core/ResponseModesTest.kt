package com.modelforge.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResponseModesTest {
    @Test
    fun normalizes_unknown_modes_to_normal() {
        assertEquals(ResponseModes.NORMAL, ResponseModes.normalize(null))
        assertEquals(ResponseModes.NORMAL, ResponseModes.normalize("turbo"))
        assertEquals(ResponseModes.INSTANT, ResponseModes.normalize("instant"))
    }

    @Test
    fun instant_caps_the_reply_and_extra_time_expands_it() {
        assertEquals(32, ResponseModes.maxNewTokens(120, ResponseModes.INSTANT))
        assertEquals(120, ResponseModes.maxNewTokens(120, ResponseModes.NORMAL))
        assertEquals(512, ResponseModes.maxNewTokens(120, ResponseModes.EXTRA))
    }

    @Test
    fun instant_skips_thinking_even_when_reasoning_is_enabled() {
        assertFalse(ResponseModes.shouldThink(true, ResponseModes.INSTANT))
        assertTrue(ResponseModes.shouldThink(true, ResponseModes.NORMAL))
        assertFalse(ResponseModes.shouldThink(false, ResponseModes.NORMAL))
    }

    @Test
    fun reasoning_prompt_is_only_added_when_thinking_runs() {
        val plain = "What is your name?"
        assertEquals(plain, ResponseModes.prompt(plain, true, ResponseModes.INSTANT))
        assertTrue(
            ResponseModes.prompt(plain, true, ResponseModes.NORMAL).contains("thinking")
        )
    }

    @Test
    fun splitThinking_separates_collapsible_reasoning() {
        val text = "Here is my answer.\n<think>The user asked a simple question.</think>\nHi, I'm Kite."
        val split = ResponseModes.splitThinking(text)
        assertEquals("The user asked a simple question.", split.thinking)
        assertEquals("Here is my answer.\n\nHi, I'm Kite.", split.answer)
    }

    @Test
    fun splitThinking_ignores_plain_text() {
        val split = ResponseModes.splitThinking("Hi, I'm Kite.")
        assertNull(split.thinking)
        assertEquals("Hi, I'm Kite.", split.answer)
    }
}
