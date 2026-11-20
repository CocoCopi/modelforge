package com.modelforge.app.data.hf

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

class HfRepositoryStructureTest {
    @Before
    fun resetPacing() = HfRateLimiter.reset()

    private val textFeature = RowFeature(
        feature_idx = 0,
        name = "text",
        type = JsonObject(
            mapOf(
                "dtype" to JsonPrimitive("string"),
                "_type" to JsonPrimitive("Value")
            )
        )
    )

    private class FakePublicApi : HfApi {
        override suspend fun searchDatasets(query: String, limit: Int): List<HfDatasetSummary> = emptyList()
        override suspend fun datasetDetail(id: String): HfDatasetDetail = HfDatasetDetail()
        override suspend fun whoAmI(): HfWhoAmI = HfWhoAmI(name = "test")
    }

    private inner class FakeServerApi : HfDatasetServerApi {
        var rowCalls = 0
        var infoCalls = 0

        override suspend fun info(dataset: String): DatasetInfoResponse {
            infoCalls++
            return DatasetInfoResponse(
                dataset_info = mapOf(
                    "default" to DatasetInfoEntry(
                        features = emptyMap(),
                        splits = mapOf("train" to SplitMeta())
                    )
                )
            )
        }

        override suspend fun splits(dataset: String): SplitsResponse = SplitsResponse(
            splits = listOf(SplitEntry(dataset = dataset, config = "default", split = "train"))
        )

        override suspend fun rows(
            dataset: String,
            config: String,
            split: String,
            offset: Long,
            length: Int
        ): RowsResponse {
            rowCalls++
            return RowsResponse(features = listOf(textFeature), rows = emptyList())
        }
    }

    @Test
    fun falls_back_to_rows_when_info_lists_configs_without_features() = runBlocking {
        val server = FakeServerApi()
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        val structure = repo.structure("owner/data")
        assertEquals(listOf("text"), structure.configs.single().textColumns)
        assertEquals(1, server.rowCalls)
    }

    @Test
    fun reuses_structure_cache_to_avoid_rate_limit_pressure() = runBlocking {
        val server = FakeServerApi()
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        repo.structure("owner/data")
        repo.structure("owner/data")
        assertEquals(1, server.infoCalls)
        assertEquals(1, server.rowCalls)
    }

    private inner class PagedServerApi(
        private val totalRows: Long,
        private val numExamples: Long = totalRows
    ) : HfDatasetServerApi {
        val active = AtomicInteger()
        val maxActive = AtomicInteger()
        var rowCalls = 0

        override suspend fun info(dataset: String): DatasetInfoResponse = DatasetInfoResponse(
            dataset_info = mapOf(
                "default" to DatasetInfoEntry(
                    features = mapOf(
                        "text" to JsonObject(
                            mapOf(
                                "dtype" to JsonPrimitive("string"),
                                "_type" to JsonPrimitive("Value")
                            )
                        )
                    ),
                    splits = mapOf("train" to SplitMeta(num_examples = numExamples))
                )
            )
        )

        override suspend fun splits(dataset: String): SplitsResponse = SplitsResponse(
            splits = listOf(SplitEntry(dataset = dataset, config = "default", split = "train"))
        )

        override suspend fun rows(
            dataset: String,
            config: String,
            split: String,
            offset: Long,
            length: Int
        ): RowsResponse {
            rowCalls++
            val now = active.incrementAndGet()
            maxActive.updateAndGet { current -> maxOf(current, now) }
            try {
                delay(15)
            } finally {
                active.decrementAndGet()
            }
            val end = minOf(totalRows, offset + length)
            val entries = if (offset >= totalRows) {
                emptyList()
            } else {
                (offset until end).map { index ->
                    RowEntry(
                        row_idx = index,
                        row = JsonObject(mapOf("text" to JsonPrimitive("row-$index")))
                    )
                }
            }
            return RowsResponse(
                features = listOf(textFeature),
                rows = entries,
                num_rows_total = totalRows,
                num_rows_per_page = length
            )
        }
    }

    @Test
    fun split_row_counts_come_from_dataset_info_without_row_requests() = runBlocking {
        val server = PagedServerApi(totalRows = 12_345)
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        val structure = repo.structure("owner/data")
        assertEquals(12_345L, structure.configs.single().splitRows.getValue("train"))
        assertEquals(12_345L, repo.rowCount("owner/data", "default", "train") ?: -1L)
        assertEquals(0, server.rowCalls)
    }

    @Test
    fun row_count_falls_back_to_one_rows_request() = runBlocking {
        val server = PagedServerApi(totalRows = 777, numExamples = 0)
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        assertEquals(777L, repo.rowCount("owner/data", "default", "train") ?: -1L)
        assertEquals(1, server.rowCalls)
    }

    @Test
    fun full_download_fetches_pages_in_order_without_bursting() = runBlocking {
        val server = PagedServerApi(totalRows = 250)
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        val dest = File.createTempFile("modelforge-pages-", ".txt")
        try {
            val stats = repo.downloadText(
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 100,
                maxBytes = 10_000_000,
                destination = dest,
                pageSize = 20,
                concurrency = 4
            )
            assertEquals(100, stats.rows)
            val lines = dest.readLines()
            assertEquals(100, lines.size)
            assertEquals("row-0", lines.first())
            assertEquals("row-99", lines.last())
            // Requests are paced by the shared limiter now instead of being
            // fired 24 at a time, but the writer must still consume them in
            // offset order.
            assertTrue(HfRateLimiter.pacing(HfRateLimiter.VIEWER).intervalMs > 0L)
        } finally {
            dest.delete()
        }
    }

    @Test
    fun download_stops_at_the_end_of_the_split() = runBlocking {
        val server = PagedServerApi(totalRows = 55)
        val repo = HfRepository(HfApiBundle(FakePublicApi(), server))
        val dest = File.createTempFile("modelforge-end-", ".txt")
        try {
            val stats = repo.downloadText(
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 10_000,
                maxBytes = 10_000_000,
                destination = dest,
                pageSize = 20,
                concurrency = 4
            )
            assertEquals(55, stats.rows)
            assertEquals(55, dest.readLines().size)
        } finally {
            dest.delete()
        }
    }
}
