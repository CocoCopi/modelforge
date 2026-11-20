package com.modelforge.app.data.hf

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class HfRateLimiterTest {

    // A controllable clock keeps these tests deterministic and instant. Real
    // sleeps are unreliable on slow CI machines (a 100 ms delay can take
    // minutes when the host is oversubscribed).
    private var now = 1_000_000L

    @Before
    fun useFakeClock() {
        HfRateLimiter.reset()
        HfRateLimiter.useClock({ now }, { millis -> now += millis })
    }

    @After
    fun restoreClock() = HfRateLimiter.reset()

    @Test
    fun requests_are_spaced_instead_of_bursting() = runBlocking {
        HfRateLimiter.acquire(HfRateLimiter.RESOLVER, authenticated = true)
        val first = HfRateLimiter.pacing(HfRateLimiter.RESOLVER)
        assertTrue("interval was ${first.intervalMs} ms", first.intervalMs >= 100L)
        assertTrue("next slot wait was ${first.waitMs} ms", first.waitMs > 0L)

        val started = now
        repeat(3) {
            HfRateLimiter.acquire(HfRateLimiter.RESOLVER, authenticated = true)
        }
        val elapsed = now - started
        // Three gaps at the 100 ms resolver floor, with at most 20% jitter.
        assertTrue("elapsed was $elapsed ms", elapsed in 300L..400L)
    }

    @Test
    fun a_rate_limit_pauses_every_worker_and_slows_the_pace() = runBlocking {
        HfRateLimiter.acquire(HfRateLimiter.HUB, authenticated = true)
        val before = HfRateLimiter.pacing(HfRateLimiter.HUB).intervalMs
        val cooldown = HfRateLimiter.noteRateLimited(
            host = HfRateLimiter.HUB,
            retryAfterSeconds = 7L,
            resetSeconds = null
        )
        assertEquals(7L, cooldown)
        assertEquals(7L, HfRateLimiter.cooldownSeconds(HfRateLimiter.HUB))
        assertTrue(HfRateLimiter.pacing(HfRateLimiter.HUB).throttled)
        assertTrue(HfRateLimiter.pacing(HfRateLimiter.HUB).intervalMs > before)
    }

    @Test
    fun acquire_waits_out_the_shared_cooldown() = runBlocking {
        HfRateLimiter.acquire(HfRateLimiter.HUB, authenticated = true)
        HfRateLimiter.noteRateLimited(
            host = HfRateLimiter.HUB,
            retryAfterSeconds = null,
            resetSeconds = null,
            minimumSeconds = 1L
        )
        val started = now
        HfRateLimiter.acquire(HfRateLimiter.HUB, authenticated = true)
        assertTrue("waited ${now - started} ms", now - started >= 1_000L)
        assertTrue(!HfRateLimiter.pacing(HfRateLimiter.HUB).throttled)
    }

    @Test
    fun rate_limit_reset_is_parsed_from_the_header() {
        assertEquals(42L, HfRateLimiter.parseResetSeconds("\"api\";r=3;t=42"))
        assertEquals(5L, HfRateLimiter.parseResetSeconds("\"api|pages|resolvers\";r=0;t=5"))
        assertEquals(9L, HfRateLimiter.parseResetSeconds("t=9"))
        assertNull(HfRateLimiter.parseResetSeconds(null))
        assertNull(HfRateLimiter.parseResetSeconds("r=3"))
    }

    @Test
    fun reset_seconds_take_precedence_over_the_minimum() = runBlocking {
        val cooldown = HfRateLimiter.noteRateLimited(
            host = HfRateLimiter.VIEWER,
            retryAfterSeconds = null,
            resetSeconds = 30L
        )
        assertEquals(30L, cooldown)
    }

    @Test
    fun a_long_reset_is_capped_so_a_download_cannot_stall_for_an_hour() = runBlocking {
        val cooldown = HfRateLimiter.noteRateLimited(
            host = HfRateLimiter.VIEWER,
            retryAfterSeconds = 9_999L,
            resetSeconds = null
        )
        assertEquals(120L, cooldown)
    }

    @Test
    fun sustained_success_slowly_recovers_the_interval() = runBlocking {
        HfRateLimiter.acquire(HfRateLimiter.VIEWER, authenticated = true)
        HfRateLimiter.noteRateLimited(HfRateLimiter.VIEWER, null, null, minimumSeconds = 1L)
        val slowed = HfRateLimiter.pacing(HfRateLimiter.VIEWER).intervalMs
        repeat(25) { HfRateLimiter.noteSuccess(HfRateLimiter.VIEWER) }
        val recovered = HfRateLimiter.pacing(HfRateLimiter.VIEWER).intervalMs
        assertTrue("recovered $recovered from $slowed", recovered < slowed)
    }
}
