package com.modelforge.app.data.ai

import com.modelforge.app.data.hf.AiDatasetSource
import com.modelforge.app.data.hf.DatasetConfigInfo
import com.modelforge.app.data.hf.DatasetStructure
import com.modelforge.app.data.hf.HfDatasetSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AiDatasetAgentTest {
    private val finalPlan = """
        {"action":"final","plan":{
          "overview":"A verified small model.",
          "training":{"steps":1200,"batch_size":10,"seq_length":128,
                      "learning_rate":0.0009,"warmup_steps":100},
          "datasets":[{"purpose":"general","dataset_id":"owner/data",
                       "config":"default","split":"train","columns":["text"],
                       "rows":1200,"weight":1.0,"license":"apache-2.0",
                       "commercial_use":true,"reason":"verified"}]
        }}
    """.trimIndent()

    @Test
    fun searches_inspects_and_returns_a_verified_plan() {
        val chat = ScriptedChat(
            mutableListOf(
                """{"action":"search","query":"stories","limit":3}""",
                """{"action":"inspect","dataset_id":"owner/data","config":"default"}""",
                finalPlan
            )
        )
        val plan = kotlinx.coroutines.runBlocking {
            AiDatasetAgent(chat, FakeSource()).design("key", request())
        }

        assertEquals(1, plan.datasets.size)
        assertEquals("owner/data", plan.datasets.single().datasetId)
        assertEquals("default", plan.datasets.single().config)
        assertEquals(listOf("text"), plan.datasets.single().columns)
        assertTrue(chat.seen.any { messages -> messages.any { it.content.contains("TOOL RESULT search") } })
        assertTrue(chat.seen.any { messages -> messages.any { it.content.contains("TOOL RESULT inspect") } })
    }

    @Test
    fun rejects_unverified_final_choices_and_retries() {
        val chat = ScriptedChat(
            mutableListOf(
                """{"action":"final","plan":{"datasets":[{"dataset_id":"owner/missing"}]}}""",
                finalPlan
            )
        )
        val plan = kotlinx.coroutines.runBlocking {
            AiDatasetAgent(chat, FakeSource()).design("key", request())
        }

        assertEquals("owner/data", plan.datasets.single().datasetId)
        assertTrue(chat.seen.size >= 2)
    }

    private fun request() = AiPlanRequest(
        name = "Kite",
        creator = "Rin",
        architecture = "gpt_text",
        parameters = 5_000_000L,
        inputs = listOf("Text"),
        outputs = listOf("Text"),
        languages = listOf("English"),
        trainingMinutes = 60,
        context = 128,
        speed = "normal",
        reasoning = false,
        toolCalling = false,
        persona = "",
        bits = 4,
        currentDatasets = emptyList()
    )

    private class ScriptedChat(private val replies: MutableList<String>) : AiChat {
        val seen = mutableListOf<List<AiMessage>>()

        override suspend fun streamChat(
            apiKey: String,
            messages: List<AiMessage>,
            onDelta: (String) -> Unit,
            onActivity: () -> Unit
        ): String {
            seen += messages.toList()
            val reply = replies.removeFirstOrNull() ?: error("No scripted reply left")
            onActivity()
            onDelta(reply)
            return reply
        }
    }

    private class FakeSource : AiDatasetSource {
        override suspend fun search(query: String, limit: Int): List<HfDatasetSummary> =
            listOf(HfDatasetSummary(id = "owner/data", downloads = 42, tags = listOf("license:apache-2.0")))

        override suspend fun detail(id: String): HfDatasetSummary =
            HfDatasetSummary(id = id, tags = listOf("license:apache-2.0"))

        override suspend fun structure(datasetId: String): DatasetStructure {
            if (datasetId != "owner/data") error("unknown dataset")
            return DatasetStructure(
                datasetId = datasetId,
                configs = listOf(DatasetConfigInfo("default", listOf("train"), listOf("text")))
            )
        }
    }
}
