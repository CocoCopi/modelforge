package com.modelforge.app.training

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CorpusBudgetTest {

    @Test
    fun steps_drive_the_budget_instead_of_a_fixed_ceiling() {
        val short = CorpusBudget.forSteps(steps = 500, batchSize = 8, seqLength = 128)
        val long = CorpusBudget.forSteps(steps = 20_000, batchSize = 8, seqLength = 128)
        assertTrue("more steps must need more text", long > short)
        assertEquals(20_000L * 8 * 128 * 3, long.toLong())
    }

    @Test
    fun budget_is_capped_by_available_ram_not_by_the_old_4m_limit() {
        val ceiling = CorpusBudget.maxCharsFor(8L * 1024 * 1024 * 1024)
        assertTrue(ceiling > 4_000_000L)
        assertTrue(ceiling <= CorpusBudget.MAX_CHARS)

        val resolved = CorpusBudget.resolve(
            requested = 0,
            steps = 100_000,
            batchSize = 8,
            seqLength = 256,
            availableRamBytes = 8L * 1024 * 1024 * 1024
        )
        assertEquals(ceiling.toInt(), resolved)
    }

    @Test
    fun honours_a_larger_requested_budget_when_ram_allows() {
        val resolved = CorpusBudget.resolve(
            requested = 12_000_000,
            steps = 100,
            batchSize = 4,
            seqLength = 128,
            availableRamBytes = 8L * 1024 * 1024 * 1024
        )
        assertEquals(12_000_000, resolved)
    }

    @Test
    fun never_returns_less_than_the_minimum() {
        val resolved = CorpusBudget.resolve(
            requested = -5,
            steps = 0,
            batchSize = 0,
            seqLength = 0,
            availableRamBytes = 0
        )
        assertEquals(CorpusBudget.MIN_CHARS.toInt(), resolved)
    }

    @Test
    fun very_low_ram_still_returns_the_minimum_budget() {
        val ceiling = CorpusBudget.maxCharsFor(2L * 1024 * 1024)
        assertEquals(CorpusBudget.MIN_CHARS, ceiling)
    }
}
