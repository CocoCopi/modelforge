package com.modelforge.app.data.hf

import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.ConcurrentHashMap
import kotlin.random.Random

/**
 * Client-side pacing for Hugging Face.
 *
 * The dataset viewer's `/rows` endpoint returns at most 100 rows per call, so a
 * 100k-row download is about 1,000 HTTPS requests. Firing 24 of those at once
 * is roughly 20x the authenticated quota (1,000 API calls per 5 minutes), so
 * the server answers with 429s. Retrying each in-flight request on its own is
 * what made the app look like it was "rate limited every 5 seconds".
 *
 * Every request now passes through one gate per host:
 *
 *  - Requests are spaced by an adaptive interval, so the app stays inside the
 *    quota instead of bursting through it.
 *  - A 429 sets a shared cooldown for the whole process, so 24 workers do not
 *    each start their own retry storm.
 *  - The cooldown comes from `Retry-After` or the `RateLimit` reset header.
 *  - The interval backs off on a 429 and recovers slowly after sustained
 *    success (additive increase, multiplicative decrease).
 *
 * This is not an arbitrary app limit: it is the server's published budget,
 * applied before the request instead of after the rejection.
 */
object HfRateLimiter {

    /** Hub API: search, dataset metadata, tree listings. */
    const val HUB = "huggingface.co"

    /** Dataset viewer: /info, /splits, /rows. */
    const val VIEWER = "datasets-server.huggingface.co"

    /** Resolver: direct data-file downloads. */
    const val RESOLVER = "huggingface.co/datasets"

    data class Pacing(
        val intervalMs: Long,
        val waitMs: Long,
        val blockedUntilMs: Long,
        private val nowMs: Long = System.currentTimeMillis()
    ) {
        val throttled: Boolean get() = blockedUntilMs > nowMs
    }

    private class Gate(
        startMs: Long,
        floorMs: Long,
        val ceilingMs: Long
    ) {
        val mutex = Mutex()
        var floorMs = floorMs
        var intervalMs = startMs
        var nextSlotMs = 0L
        var blockedUntilMs = 0L
        var successes = 0
    }

    private val gates = ConcurrentHashMap<String, Gate>()

    /**
     * Time source. Tests replace it so pacing can be verified without real
     * sleeps (some CI machines stretch a 100 ms delay into minutes).
     */
    @Volatile
    private var clock: () -> Long = System::currentTimeMillis

    @Volatile
    private var sleeper: suspend (Long) -> Unit = { delay(it) }

    internal fun useClock(now: () -> Long, sleep: suspend (Long) -> Unit) {
        clock = now
        sleeper = sleep
    }

    /**
     * The Hub allows 1,000 API calls per 5 minutes for a signed-in user, so one
     * call every 300 ms is the hard ceiling. 380 ms leaves room for metadata
     * and the AI's own search/inspect calls. Anonymous users get 500 per 5
     * minutes.
     *
     * The viewer does not publish its quota. Start below the Hub rate and let
     * the additive-increase step find the real ceiling; a 429 immediately
     * halves the pace and pauses every worker together.
     */
    private fun floorMs(host: String, authenticated: Boolean): Long = when (host) {
        HUB -> if (authenticated) 380L else 700L
        VIEWER -> if (authenticated) 120L else 220L
        else -> if (authenticated) 100L else 180L
    }

    private fun startMs(host: String, authenticated: Boolean): Long = when (host) {
        HUB -> floorMs(host, authenticated)
        VIEWER -> if (authenticated) 250L else 350L
        else -> if (authenticated) 150L else 250L
    }

    private fun gate(host: String, authenticated: Boolean): Gate =
        gates.computeIfAbsent(host) {
            Gate(
                startMs = startMs(host, authenticated),
                floorMs = floorMs(host, authenticated),
                ceilingMs = 5_000L
            )
        }

    /**
     * Waits until this request may be sent. The mutex is only held to reserve a
     * slot; the actual wait happens outside it, so a 429 can shorten or extend
     * the cooldown for everybody immediately.
     */
    suspend fun acquire(host: String = HUB, authenticated: Boolean = false) {
        val state = gate(host, authenticated)
        val floor = floorMs(host, authenticated)
        while (true) {
            val waitMs = state.mutex.withLock {
                if (state.intervalMs < floor) state.intervalMs = floor
                state.floorMs = floor
                val now = clock()
                val blocked = state.blockedUntilMs - now
                if (blocked > 0) {
                    blocked
                } else {
                    val slot = maxOf(state.nextSlotMs, now)
                    val jitter = if (state.intervalMs >= 50L) {
                        Random.nextLong(0L, state.intervalMs / 5L + 1L)
                    } else {
                        0L
                    }
                    state.nextSlotMs = slot + state.intervalMs + jitter
                    slot - now
                }
            }
            if (waitMs <= 0L) return
            sleeper(waitMs)
        }
    }

    suspend fun noteSuccess(host: String) {
        val state = gates[host] ?: return
        state.mutex.withLock {
            state.successes++
            if (state.successes >= 25) {
                state.successes = 0
                state.intervalMs = maxOf(state.floorMs, state.intervalMs * 9L / 10L)
            }
        }
    }

    suspend fun noteBusy(host: String, seconds: Long = 2L) {
        val state = gates[host] ?: return
        state.mutex.withLock {
            val until = clock() + seconds.coerceIn(1L, 30L) * 1_000L
            state.blockedUntilMs = maxOf(state.blockedUntilMs, until)
            state.nextSlotMs = maxOf(state.nextSlotMs, state.blockedUntilMs)
        }
    }

    /**
     * Called on HTTP 429. Returns the cooldown in seconds so the caller can tell
     * the user how long it will be.
     */
    suspend fun noteRateLimited(
        host: String,
        retryAfterSeconds: Long?,
        resetSeconds: Long?,
        minimumSeconds: Long = 5L
    ): Long {
        val state = gate(host, false)
        val cooldown = maxOf(retryAfterSeconds ?: 0L, resetSeconds ?: 0L, minimumSeconds)
            .coerceIn(1L, 120L)
        state.mutex.withLock {
            val until = clock() + cooldown * 1_000L
            state.blockedUntilMs = maxOf(state.blockedUntilMs, until)
            state.intervalMs = minOf(state.ceilingMs, maxOf(state.floorMs, state.intervalMs * 2L))
            state.successes = 0
            state.nextSlotMs = maxOf(state.nextSlotMs, state.blockedUntilMs)
        }
        return cooldown
    }

    fun pacing(host: String = HUB): Pacing {
        val state = gates[host] ?: return Pacing(0L, 0L, 0L)
        val now = clock()
        val blocked = state.blockedUntilMs - now
        val wait = maxOf(blocked, state.nextSlotMs - now, 0L)
        return Pacing(state.intervalMs, wait, state.blockedUntilMs, now)
    }

    /** Seconds until the current cooldown ends, or 0 when not throttled. */
    fun cooldownSeconds(host: String = HUB): Long {
        val blocked = pacing(host).blockedUntilMs - clock()
        return if (blocked <= 0L) 0L else (blocked + 999L) / 1_000L
    }

    /** Reads `RateLimit: "api";r=12;t=42` and returns the reset seconds. */
    fun parseResetSeconds(header: String?): Long? {
        if (header.isNullOrBlank()) return null
        return Regex("""(?:^|[;,\s])t=(\d+)""").find(header)?.groupValues?.get(1)?.toLongOrNull()
    }

    /** Clears pacing state and restores the real clock. */
    fun reset() {
        gates.clear()
        clock = System::currentTimeMillis
        sleeper = { delay(it) }
    }
}
