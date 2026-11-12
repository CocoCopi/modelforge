package com.modelforge.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ConfigDeriverTest {

    private fun assertTargetWithin5Percent(target: Long) {
        val result = ConfigDeriver.derive(target)
        val cfg = result.config
        val count = cfg.paramCount()
        val err = kotlin.math.abs(count - target).toDouble() / target.toDouble()
        assertTrue(
            "target=$target count=$count err=$err config=$cfg",
            err <= 0.05
        )
        assertTrue("architecture must be valid: $cfg", cfg.isValid())
        assertEquals(0, cfg.dModel % cfg.heads)
        assertTrue("head dim should be sane: ${cfg.headDim}", cfg.headDim in 16..64)
        assertTrue("must have at least one head", cfg.heads >= 1)
        assertTrue("layers should be positive", cfg.layers >= 1)
        assertTrue("vocab should be non-trivial", cfg.vocabSize >= 1024)
        assertTrue("context should be usable", cfg.contextLength >= 128)
        assertEquals(target, cfg.targetParams)
        assertTrue("result should report tolerance", result.withinTolerance)
    }

    @Test
    fun derivesOneMillionWithinFivePercent() = assertTargetWithin5Percent(1_000_000L)

    @Test
    fun derivesTenMillionWithinFivePercent() = assertTargetWithin5Percent(10_000_000L)

    @Test
    fun derivesFiftyMillionWithinFivePercent() = assertTargetWithin5Percent(50_000_000L)

    @Test
    fun respectsDeviceParameterCap() {
        val cap = 3_000_000L
        val result = ConfigDeriver.derive(50_000_000L, ConfigDeriver.Options(maxParams = cap))
        assertTrue(result.config.paramCount() <= cap)
        assertTrue(result.explanation.contains("capped", ignoreCase = true))
    }

    @Test
    fun parameterCountFormulaIsStable() {
        val cfg = ModelConfig(
            vocabSize = 1024,
            contextLength = 128,
            dModel = 64,
            layers = 2,
            heads = 4,
            dFf = 256
        )
        // V*d + C*d + L*(4d^2 + 2*d*df + 9d + df) + 2d
        val expected = 1024L * 64 + 128L * 64 + 2L * (4L * 64 * 64 + 2L * 64 * 256 + 9L * 64 + 256) + 2L * 64
        assertEquals(expected, cfg.paramCount())
    }

    @Test
    fun packingShrinksWithFewerBits() {
        val result = ConfigDeriver.derive(10_000_000L)
        val cfg = result.config
        val fp32 = cfg.fp32Bytes()
        val b8 = cfg.packedExportBytes(8)
        val b4 = cfg.packedExportBytes(4)
        val b2 = cfg.packedExportBytes(2)
        assertTrue(b8 < fp32)
        assertTrue(b4 < b8)
        assertTrue(b2 < b4)
        assertTrue(cfg.packedSavingsVsFp32(4) > 0.5)
    }
}
