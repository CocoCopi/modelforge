package com.modelforge.app.data.hf

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Test

class HfRepositoryStructureTest {
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
}
