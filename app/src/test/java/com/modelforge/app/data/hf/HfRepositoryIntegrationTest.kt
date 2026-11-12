package com.modelforge.app.data.hf

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Hits the real Hugging Face APIs. This is deliberately an integration test: it
 * proves search, metadata, splits and page downloads work against real data.
 */
class HfRepositoryIntegrationTest {

    private val repo = HfRepository()

    private suspend fun <T> retry(attempts: Int = 3, block: suspend () -> T): T {
        var last: Throwable? = null
        repeat(attempts) { attempt ->
            try {
                return block()
            } catch (t: Throwable) {
                last = t
                if (attempt < attempts - 1) delay(2_000)
            }
        }
        throw last ?: IllegalStateException("retry failed")
    }

    @Test
    fun searchAndDownloadRealDataset() = runBlocking {
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
                    datasetId = "roneneldan/TinyStories",
                    config = "default",
                    split = "train",
                    columns = listOf("text"),
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
