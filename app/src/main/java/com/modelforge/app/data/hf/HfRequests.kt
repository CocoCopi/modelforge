package com.modelforge.app.data.hf

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import retrofit2.HttpException
import java.io.IOException

/**
 * Runs one Hugging Face request through [HfRateLimiter] and retries a 429 or a
 * transient failure. A 429 does not sleep here: it sets the shared cooldown, so
 * every worker waits together instead of each firing its own retry.
 */
internal suspend fun <T> pacedRequest(
    host: String,
    authenticated: Boolean,
    maxAttempts: Int = 3,
    onThrottle: (String) -> Unit = {},
    block: suspend () -> T
): T {
    var attempt = 0
    while (true) {
        try {
            HfRateLimiter.acquire(host, authenticated)
            val value = block()
            HfRateLimiter.noteSuccess(host)
            return value
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            val code = (t as? HttpException)?.code()
            val rateLimited = code == 429
            val transient = t is IOException || rateLimited ||
                code == 500 || code == 502 || code == 503 || code == 504
            if (!transient || attempt >= maxAttempts - 1) throw t
            if (rateLimited) {
                val headers = (t as? HttpException)?.response()?.headers()
                val cooldown = HfRateLimiter.noteRateLimited(
                    host = host,
                    retryAfterSeconds = headers?.get("Retry-After")?.toLongOrNull(),
                    resetSeconds = HfRateLimiter.parseResetSeconds(headers?.get("RateLimit"))
                )
                onThrottle(
                    "Hugging Face asked us to slow down; resuming in ${cooldown}s " +
                        "(requests stay inside your plan's quota)"
                )
            } else {
                HfRateLimiter.noteBusy(host)
                delay(1_000L * (attempt + 1))
            }
            attempt++
        }
    }
}
