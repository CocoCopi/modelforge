package com.modelforge.app.training

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class RecoveryPolicyTest {
    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun interval_scales_with_model_size() {
        assertEquals(5, RecoveryPolicy.intervalFor(5_000_000L))
        assertEquals(10, RecoveryPolicy.intervalFor(20_000_000L))
        assertEquals(20, RecoveryPolicy.intervalFor(50_000_000L))
    }

    @Test
    fun checkpoints_are_due_on_interval_and_at_the_end() {
        assertTrue(RecoveryPolicy.isDue(5, 500, 5))
        assertFalse(RecoveryPolicy.isDue(6, 500, 5))
        assertTrue(RecoveryPolicy.isDue(500, 500, 5))
    }

    @Test
    fun checkpoint_header_reads_the_little_endian_step() {
        val file = File(temp.root, "recovery.mfck")
        val bytes = ByteArray(64)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).apply {
            putInt(0x4D46434B)
            putLong(36, 78L)
        }
        file.writeBytes(bytes)
        assertEquals(78L, CheckpointHeader.readStep(file))
        assertTrue(CheckpointHeader.hasMagic(file))
    }

    @Test
    fun missing_or_short_checkpoint_has_no_step() {
        val file = File(temp.root, "empty.mfck")
        assertNull(CheckpointHeader.readStep(file))
        file.writeBytes(ByteArray(8))
        assertNull(CheckpointHeader.readStep(file))
        assertFalse(CheckpointHeader.hasMagic(file))
    }
}
