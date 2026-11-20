package com.modelforge.app.data.hf

import kotlinx.coroutines.runBlocking
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * Proves the downloader takes the repository-file path when one exists, so a
 * bulk download no longer spends one viewer request per 100 rows.
 */
class HfRepositoryDirectTest {

    @Before
    fun resetPacing() = HfRateLimiter.reset()

    private class FakePublicApi : HfApi {
        override suspend fun searchDatasets(query: String, limit: Int) = emptyList<HfDatasetSummary>()
        override suspend fun datasetDetail(id: String) = HfDatasetDetail(id = id)
        override suspend fun whoAmI() = HfWhoAmI(name = "test")
    }

    private class CountingServerApi : HfDatasetServerApi {
        val rowCalls = AtomicInteger()

        override suspend fun info(dataset: String) = DatasetInfoResponse()
        override suspend fun splits(dataset: String) = SplitsResponse()
        override suspend fun rows(
            dataset: String,
            config: String,
            split: String,
            offset: Long,
            length: Int
        ): RowsResponse {
            rowCalls.incrementAndGet()
            return RowsResponse(
                features = listOf(
                    RowFeature(0, "text", kotlinx.serialization.json.JsonObject(emptyMap()))
                ),
                rows = List(length) { index ->
                    RowEntry(
                        row_idx = offset + index,
                        row = kotlinx.serialization.json.JsonObject(
                            mapOf(
                                "text" to kotlinx.serialization.json.JsonPrimitive("viewer $index")
                            )
                        )
                    )
                },
                num_rows_total = 10_000,
                num_rows_per_page = length
            )
        }
    }

    private class FakeFileApi(
        private val listings: Map<String, List<HfTreeEntry>>,
        private val files: Map<String, String>
    ) : HfFileApi {
        val fileCalls = AtomicInteger()

        override suspend fun treeRoot(id: String, revision: String) =
            listings[""] ?: emptyList()

        override suspend fun treeDir(id: String, revision: String, path: String) =
            listings[path] ?: emptyList()

        override suspend fun file(
            url: String,
            range: String?
        ): ResponseBody {
            fileCalls.incrementAndGet()
            val path = url.substringAfter("/resolve/main/")
            return (files[path] ?: "").toResponseBody("text/plain".toMediaType())
        }
    }

    @Test
    fun download_uses_the_data_file_instead_of_the_viewer() = runBlocking {
        val server = CountingServerApi()
        val fileApi = FakeFileApi(
            listings = mapOf(
                "" to listOf(HfTreeEntry("file", "train.jsonl", 100_000))
            ),
            files = mapOf(
                "train.jsonl" to buildString {
                    repeat(5) { append("""{"text":"direct $it"}""").append('\n') }
                }
            )
        )
        val repo = HfRepository(
            HfApiBundle(FakePublicApi(), server, fileApi, { true })
        )
        val dest = File.createTempFile("modelforge-fast-", ".txt")
        try {
            val stats = repo.downloadText(
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 4,
                maxBytes = 1_000_000,
                destination = dest,
                knownTotalRows = 5
            )
            assertEquals(4L, stats.rows)
            assertEquals(0, server.rowCalls.get())
            assertEquals(1, fileApi.fileCalls.get())
            assertTrue(dest.readText().contains("direct 0"))
        } finally {
            dest.delete()
        }
    }

    @Test
    fun download_falls_back_to_the_viewer_without_a_text_file() = runBlocking {
        val server = CountingServerApi()
        val fileApi = FakeFileApi(
            listings = mapOf(
                "" to listOf(HfTreeEntry("file", "train.parquet", 100_000))
            ),
            files = emptyMap()
        )
        val repo = HfRepository(
            HfApiBundle(FakePublicApi(), server, fileApi, { true })
        )
        val dest = File.createTempFile("modelforge-fallback-", ".txt")
        try {
            val stats = repo.downloadText(
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 3,
                maxBytes = 1_000_000,
                destination = dest,
                knownTotalRows = 10_000
            )
            assertEquals(3L, stats.rows)
            assertTrue(server.rowCalls.get() > 0)
        } finally {
            dest.delete()
        }
    }
}
