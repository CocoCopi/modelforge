package com.modelforge.app.data.hf

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Hits the real Hugging Face APIs. This is deliberately an integration test: it
 * proves search, metadata, splits and page downloads work against real data.
 *
 * The Hub answers shared CI addresses with 429s from time to time. Waiting out
 * a rate-limit window would stall the whole build for minutes, so the test
 * runs on a controllable clock (a throttled window elapses at once) and skips
 * instead of failing when the runner is throttled.
 */
class HfRepositoryIntegrationTest {

    private val repo = HfRepository()
    private var fakeNow = System.currentTimeMillis()

    @Before
    fun installFastClock() {
        fakeNow = System.currentTimeMillis()
        HfRateLimiter.reset()
        HfRateLimiter.useClock(
            { fakeNow },
            { millis ->
                fakeNow += millis
                // Keep the pacing real but bounded: a five-minute window must
                // not hold the build hostage.
                delay(minOf(millis, 50L))
            }
        )
    }

    @After
    fun restoreClock() = HfRateLimiter.reset()

    private fun isRateLimited(t: Throwable): Boolean {
        var current: Throwable? = t
        while (current != null) {
            if (current.message?.contains("429", ignoreCase = true) == true) return true
            if (current is HfException && current.message?.contains("rate limit", ignoreCase = true) == true) {
                return true
            }
            current = current.cause
        }
        return false
    }

    private suspend fun <T> retry(attempts: Int = 3, block: suspend () -> T): T {
        var last: Throwable? = null
        repeat(attempts) { attempt ->
            try {
                return block()
            } catch (t: Throwable) {
                last = t
                // A rate limit is not a transient hiccup: the window is five
                // minutes long, so retrying only makes the build slower.
                if (isRateLimited(t)) throw t
                if (attempt < attempts - 1) delay(2_000)
            }
        }
        throw last ?: IllegalStateException("retry failed")
    }

    @Test
    fun searchAndDownloadRealDataset() = runBlocking {
        try {
            runSearchAndDownload()
        } catch (t: Throwable) {
            if (isRateLimited(t)) {
                assumeNoException("Hugging Face rate limited this runner", t)
            }
            throw t
        }
    }

    private suspend fun runSearchAndDownload() {
        val results = retry { repo.search("tinystories", 8) }
        assertTrue("search returned results", results.isNotEmpty())
        assertTrue(
            "expected TinyStories in search results",
            results.any { it.id.contains("TinyStories", ignoreCase = true) }
        )

        val detail = retry { repo.detail("roneneldan/TinyStories") }
        val license = detail.resolvedLicense()
        assertNotNull("license should be discoverable", license)
        assertTrue("TinyStories license is not flagged non-commercial", !isLikelyNonCommercial(license))

        val structure = retry { repo.structure("roneneldan/TinyStories") }
        val config = structure.configs.firstOrNull { it.name == "default" }
            ?: error("default config exists")
        assertTrue("train split exists", config.splits.contains("train"))
        assertTrue("text column is discovered", config.textColumns.contains("text"))

        val dest = File.createTempFile("modelforge-hf-", ".txt")
        try {
            val stats = retry {
                repo.downloadText(
                    datasetId = "databricks/databricks-dolly-15k",
                    config = "default",
                    split = "train",
                    columns = listOf("instruction", "response"),
                    maxRows = 3,
                    maxBytes = 200_000,
                    destination = dest
                )
            }
            assertTrue("downloaded rows", stats.rows in 1..3)
            assertTrue("downloaded bytes", stats.bytes > 0)
            assertTrue("file has content", dest.length() > 0)
            assertTrue("preview has letters", stats.preview.any { it.isLetter() })
            val text = dest.readText()
            assertTrue("cached text has letters", text.any { it.isLetter() })
        } finally {
            dest.delete()
        }
    }

    @Test
    fun licenseWarningRecognisesNonCommercial() {
        assertTrue(isLikelyNonCommercial("cc-by-nc-4.0"))
        assertTrue(isLikelyNonCommercial("unknown"))
        assertTrue(isLikelyNonCommercial(null))
        assertTrue(!isLikelyNonCommercial("apache-2.0"))
        assertTrue(!isLikelyNonCommercial("mit"))
        assertTrue(!isLikelyNonCommercial("cdla-sharing-1.0"))
    }
}
