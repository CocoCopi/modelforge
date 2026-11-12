package com.modelforge.app.training

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class IdentityDataGeneratorTest {
    @Test
    fun generates_at_least_300_varied_conversations() {
        val corpus = IdentityDataGenerator.generate(
            name = "Pocket Forge",
            creator = "Ada",
            languages = listOf("English", "Spanish")
        )
        assertTrue(corpus.conversations.size >= 300)
        assertTrue(corpus.text.length > 10_000)
        assertTrue(corpus.conversations.map { it.text }.toSet().size > 50)
    }

    @Test
    fun every_conversation_names_the_model_and_creator() {
        val corpus = IdentityDataGenerator.generate("Pocket Forge", "Ada", listOf("English"))
        corpus.conversations.forEach { conversation ->
            assertTrue(conversation.text.contains("Pocket Forge"))
            assertTrue(conversation.text.contains("Ada"))
        }
    }

    @Test
    fun selected_languages_are_all_present() {
        val languages = listOf("English", "French", "Japanese")
        val corpus = IdentityDataGenerator.generate("Kite", "Rin", languages)
        val present = corpus.conversations.map { it.language }.toSet()
        languages.forEach { assertTrue(it, it in present) }
    }

    @Test
    fun never_claims_to_be_another_model() {
        val corpus = IdentityDataGenerator.generate("Kite", "Rin", listOf("English"))
        assertFalse(corpus.text.contains("I am ChatGPT"))
        assertFalse(corpus.text.contains("I'm ChatGPT"))
    }

    @Test
    fun respects_a_character_budget() {
        val corpus = IdentityDataGenerator.generate(
            "Kite",
            "Rin",
            listOf("English"),
            maxChars = 2_000
        )
        assertTrue(corpus.text.length <= 2_000)
        assertTrue(corpus.conversations.isNotEmpty())
    }

    @Test
    fun identity_fraction_stays_in_the_requested_band() {
        val budget = IdentityDataGenerator.estimateIdentityFraction(1_500_000)
        assertEquals(60_000, budget)
        assertTrue(budget.toDouble() / 1_500_000.0 in 0.03..0.05)
    }

    @Test
    fun system_prompt_names_the_creator() {
        val prompt = IdentityDataGenerator.systemPrompt("Kite", "Rin")
        assertTrue(prompt.contains("Kite"))
        assertTrue(prompt.contains("Rin"))
        assertTrue(prompt.contains("small language model"))
    }
}
