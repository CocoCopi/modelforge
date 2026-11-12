package com.modelforge.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ArchitectureRegistryTest {
    @Test
    fun gpt_is_available_and_derives_a_counted_config() {
        val info = ArchitectureRegistry.info(ArchitectureRegistry.GPT_TEXT)
        assertTrue(info.available)
        assertEquals(ArchitectureStatus.STABLE, info.status)
        assertEquals(listOf(Modality.TEXT), info.inputs)
        assertEquals(listOf(Modality.TEXT), info.outputs)

        val result = ArchitectureRegistry.derive(
            ArchitectureRegistry.GPT_TEXT,
            targetParams = 5_000_000L,
            maxParams = 50_000_000L
        )
        val config = result.config
        assertNotNull(config)
        assertEquals(ArchitectureRegistry.GPT_TEXT, config!!.architectureId)
        assertTrue(config.paramCount() > 0L)
        assertTrue(result.errorFraction < 0.05)
    }

    @Test
    fun planned_architectures_are_visible_but_not_selectable() {
        val planned = ArchitectureRegistry.all().filter { it.status == ArchitectureStatus.PLANNED }
        assertTrue(planned.size >= 8)
        planned.forEach { info ->
            assertFalse(info.available)
            assertNotNull(info.unavailableReason)
            assertTrue(info.supportedBits.isEmpty())
            val result = ArchitectureRegistry.derive(info.id, 5_000_000L, 50_000_000L)
            assertEquals(null, result.config)
        }
    }

    @Test
    fun ids_are_unique_and_all_modalities_are_declared() {
        val all = ArchitectureRegistry.all()
        assertEquals(all.size, all.map { it.id }.toSet().size)
        all.forEach { info ->
            assertTrue(info.inputs.isNotEmpty())
            assertTrue(info.outputs.isNotEmpty())
            assertTrue(info.displayName.isNotBlank())
            assertTrue(info.summary.isNotBlank())
        }
    }

    @Test
    fun gpt_supports_the_packed_export_bits() {
        assertEquals(listOf(2, 3, 4, 8, 16, 32), ArchitectureRegistry.supportedBits(ArchitectureRegistry.GPT_TEXT))
    }

    @Test
    fun unknown_architecture_falls_back_to_stable_gpt() {
        assertEquals(ArchitectureRegistry.GPT_TEXT, ArchitectureRegistry.info("missing").id)
    }
}
