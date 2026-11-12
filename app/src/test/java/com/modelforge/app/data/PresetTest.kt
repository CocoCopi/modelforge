package com.modelforge.app.data

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PresetTest {
    @Test
    fun preset_round_trips_through_json_with_defaults() {
        val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }
        val preset = ModelPreset(
            id = "preset-1",
            name = "Tiny storyteller",
            targetParams = 5_000_000L,
            languages = listOf("English", "Spanish"),
            reasoningEnabled = true,
            reasoningRoom = 384,
            steps = 700
        )
        val decoded = json.decodeFromString<ModelPreset>(json.encodeToString(preset))
        assertEquals(preset, decoded)
    }

    @Test
    fun summary_describes_the_saved_shape() {
        val preset = ModelPreset(id = "p", name = "Quick", targetParams = 10_000_000L, bits = 4)
        assertTrue(preset.summary.contains("10M"))
        assertTrue(preset.summary.contains("4-bit"))
        assertTrue(preset.summary.contains("500 steps"))
    }
}
