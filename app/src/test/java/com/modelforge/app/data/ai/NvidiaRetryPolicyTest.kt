package com.modelforge.app.data.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NvidiaRetryPolicyTest {
    @Test
    fun retries_server_errors_and_rate_limits_only() {
        assertTrue(NvidiaRetryPolicy.isRetryable(429))
        assertTrue(NvidiaRetryPolicy.isRetryable(500))
        assertTrue(NvidiaRetryPolicy.isRetryable(503))
        assertFalse(NvidiaRetryPolicy.isRetryable(401))
        assertFalse(NvidiaRetryPolicy.isRetryable(404))
    }

    @Test
    fun uses_retry_after_when_present() {
        assertEquals(42L, NvidiaRetryPolicy.waitSeconds(503, 42L, 0))
        assertEquals(120L, NvidiaRetryPolicy.waitSeconds(503, 999L, 0))
        assertEquals(1L, NvidiaRetryPolicy.waitSeconds(503, 0L, 0))
    }

    @Test
    fun backs_off_for_server_and_rate_limit_errors() {
        assertEquals(5L, NvidiaRetryPolicy.waitSeconds(503, null, 0))
        assertEquals(15L, NvidiaRetryPolicy.waitSeconds(503, null, 1))
        assertEquals(30L, NvidiaRetryPolicy.waitSeconds(503, null, 2))
        assertEquals(20L, NvidiaRetryPolicy.waitSeconds(429, null, 0))
        assertEquals(45L, NvidiaRetryPolicy.waitSeconds(429, null, 1))
        assertEquals(60L, NvidiaRetryPolicy.waitSeconds(429, null, 2))
    }
}
