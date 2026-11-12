package com.modelforge.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class IdentityCheckTest {
    @Test
    fun requires_both_name_and_creator_case_insensitively() {
        assertTrue(IdentityCheck.isIdentityAnswer("Kite", "Rin", "I'm Kite, created by Rin."))
        assertTrue(IdentityCheck.isIdentityAnswer("Kite", "Rin", "kite was made by RIN"))
        assertFalse(IdentityCheck.isIdentityAnswer("Kite", "Rin", "I'm Kite."))
        assertFalse(IdentityCheck.isIdentityAnswer("Kite", "Rin", "I was made by Rin."))
        assertFalse(IdentityCheck.isIdentityAnswer("Kite", "Rin", ""))
    }

    @Test
    fun passes_at_eighty_percent_and_offers_reinforce_below_it() {
        val responses = List(20) { index ->
            if (index < 16) "I'm Kite, created by Rin." else "I don't know."
        }
        val result = IdentityCheck.evaluate("Kite", "Rin", responses)
        assertEquals(16, result.passed)
        assertEquals(20, result.total)
        assertFalse(result.needsReinforce)

        val weak = IdentityCheck.evaluate("Kite", "Rin", List(20) { "I don't know." })
        assertTrue(weak.needsReinforce)
    }

    @Test
    fun generates_twenty_questions_across_selected_languages() {
        val questions = IdentityCheck.questions(listOf("English", "Spanish"))
        assertEquals(20, questions.size)
        assertTrue(questions.any { it.contains("¿") })
        assertTrue(questions.any { it.contains("?") })
    }
}
