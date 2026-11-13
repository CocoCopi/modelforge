package com.modelforge.app.data.ai

import com.modelforge.app.data.hf.DatasetConfigInfo
import com.modelforge.app.data.hf.DatasetStructure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AiPlannerTest {
    private fun request() = AiPlanRequest(
        name = "Kite",
        creator = "Rin",
        architecture = "gpt_text",
        parameters = 5_000_000L,
        inputs = listOf("Text"),
        outputs = listOf("Text"),
        languages = listOf("English", "Tamil"),
        trainingMinutes = 90,
        context = 128,
        speed = "normal",
        reasoning = true,
        toolCalling = true,
        persona = "Be honest.",
        bits = 4,
        currentDatasets = emptyList()
    )

    @Test
    fun prompt_includes_tamil_modalities_and_commercial_licence_rules() {
        val system = AiPlanner.systemPrompt()
        val user = AiPlanner.userPrompt(request())
        assertTrue(system.contains("commercial-use"))
        assertTrue(system.contains("Tamil"))
        assertTrue(system.contains("gpt_text"))
        assertTrue(system.contains("ncm_text"))
        assertTrue(system.contains("ai_architecture"))
        assertTrue(user.contains("Tamil"))
        assertTrue(user.contains("training_minutes: 90"))
        assertTrue(user.contains("inputs: Text"))
        assertTrue(user.contains("reasoning: true"))
        assertTrue(system.contains("most fluent"))
        assertTrue(system.contains("800-1600 steps"))
        assertTrue(system.contains("do not invent ids"))
    }

    @Test
    fun parses_a_fenced_plan_and_sanitizes_values() {
        val raw = """
            Here is the design.
            ```json
            {
              "overview": "A tiny kite model.",
              "analogy": "Like a paper kite.",
              "training": {"steps": 9000, "batch_size": 99, "seq_length": 12,
                           "learning_rate": 0.5, "warmup_steps": 0},
              "datasets": [
                {"purpose": "general", "dataset_id": "owner/data", "config": "default",
                 "split": "train", "columns": ["text"], "rows": 99999, "weight": 9,
                 "license": "apache-2.0", "commercial_use": true, "reason": "clean"},
                {"purpose": "general", "dataset_id": "owner/data", "commercial_use": true}
              ],
              "notes": ["keep it small"],
              "warnings": []
            }
            ```
        """.trimIndent()
        val plan = AiPlanner.parse(raw)
        assertEquals("A tiny kite model.", plan.overview)
        assertEquals(2000, plan.training.steps)
        assertEquals(32, plan.training.batchSize)
        assertEquals(32, plan.training.seqLength)
        assertEquals(1, plan.datasets.size)
        assertEquals(5000, plan.datasets.first().rows)
        assertEquals(5f, plan.datasets.first().weight, 1e-6f)
    }

    @Test
    fun keeps_the_same_dataset_id_when_configs_differ() {
        val plan = AiPlanner.sanitize(
            AiPlan(
                overview = "x",
                datasets = listOf(
                    AiDatasetSuggestion(datasetId = "owner/data", config = "sample-10BT"),
                    AiDatasetSuggestion(datasetId = "owner/data", config = "sample-100BT")
                )
            )
        )
        assertEquals(2, plan.datasets.size)
    }

    @Test
    fun flags_non_commercial_or_unknown_datasets() {
        val plan = AiPlanner.sanitize(
            AiPlan(
                overview = "x",
                datasets = listOf(
                    AiDatasetSuggestion(
                        datasetId = "owner/nc",
                        commercialUse = false,
                        license = "cc-by-nc-4.0"
                    )
                )
            )
        )
        assertTrue(plan.warnings.any { it.contains("commercial use is not confirmed") })
    }

    @Test
    fun parses_plain_json_without_markdown() {
        val plan = AiPlanner.parse("""{"overview":"plain","datasets":[]}""")
        assertEquals("plain", plan.overview)
        assertTrue(plan.datasets.isEmpty())
    }

    @Test
    fun stream_parser_reads_openai_style_delta() {
        val line = """{"choices":[{"delta":{"content":"hello"}}]}"""
        assertEquals("hello", NvidiaStreamParser.delta(line))
        assertEquals("jo", NvidiaStreamParser.delta("""{"choices":[{"delta":{"content":[{"type":"text","text":"jo"}]}}]}"""))
    }
}

class NvidiaResponseTest {
    @Test
    fun parses_non_streaming_message_content() {
        val raw = """{"choices":[{"message":{"role":"assistant","content":"{\"overview\":\"ok\"}"}}]}"""
        assertEquals("{\"overview\":\"ok\"}", NvidiaStreamParser.fullContent(raw))
    }

    @Test
    fun ignores_reasoning_frames_for_content_but_reads_them_separately() {
        val reasoning = """{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"""
        assertEquals(null, NvidiaStreamParser.delta(reasoning))
        assertEquals("thinking", NvidiaStreamParser.reasoningDelta(reasoning))
    }
}

class AiDatasetResolverTest {
    private fun structure() = DatasetStructure(
        datasetId = "owner/data",
        configs = listOf(
            DatasetConfigInfo("empty", listOf("train"), emptyList()),
            DatasetConfigInfo("default", listOf("train", "validation"), listOf("prompt", "response")),
            DatasetConfigInfo("other", listOf("test"), listOf("text"))
        )
    )

    @Test
    fun uses_requested_config_split_and_columns_when_they_exist() {
        val resolved = AiDatasetResolver.resolve(
            structure(),
            AiDatasetSuggestion(
                datasetId = "owner/data",
                config = "default",
                split = "validation",
                columns = listOf("prompt", "missing")
            )
        )
        assertEquals("default", resolved?.config)
        assertEquals("validation", resolved?.split)
        assertEquals(listOf("prompt"), resolved?.columns)
    }

    @Test
    fun rejects_a_hallucinated_explicit_config_instead_of_silently_using_another() {
        val resolved = AiDatasetResolver.resolve(
            structure(),
            AiDatasetSuggestion(datasetId = "owner/data", config = "missing")
        )
        assertEquals(null, resolved)
    }

    @Test
    fun falls_back_to_a_real_text_config_and_split_when_config_is_blank() {
        val resolved = AiDatasetResolver.resolve(
            structure(),
            AiDatasetSuggestion(datasetId = "owner/data", config = "", split = "missing")
        )
        assertEquals("default", resolved?.config)
        assertEquals("train", resolved?.split)
        assertEquals(listOf("prompt", "response"), resolved?.columns)
    }

    @Test
    fun returns_null_when_no_text_columns_exist() {
        val onlyEmpty = DatasetStructure(
            datasetId = "owner/data",
            configs = listOf(DatasetConfigInfo("default", listOf("train"), emptyList()))
        )
        assertEquals(null, AiDatasetResolver.resolve(onlyEmpty, AiDatasetSuggestion(datasetId = "owner/data")))
    }
}


class AiPlanDeduplicationTest {
    @Test
    fun keeps_the_same_dataset_with_different_configs() {
        val plan = AiPlanner.sanitize(
            AiPlan(
                datasets = listOf(
                    AiDatasetSuggestion(datasetId = "owner/data", config = "english", split = "train"),
                    AiDatasetSuggestion(datasetId = "owner/data", config = "tamil", split = "train")
                )
            )
        )
        assertEquals(2, plan.datasets.size)
    }
}

class AiDatasetColumnQualityTest {
    @Test
    fun prefers_content_columns_over_system_metadata() {
        val structure = DatasetStructure(
            datasetId = "owner/chat",
            configs = listOf(DatasetConfigInfo("default", listOf("train"), listOf("system", "chat")))
        )
        val resolved = AiDatasetResolver.resolve(
            structure,
            AiDatasetSuggestion(datasetId = "owner/chat", config = "default")
        )
        assertEquals(listOf("chat"), resolved?.columns)
    }

    @Test
    fun rejects_a_dataset_with_only_metadata_text() {
        val structure = DatasetStructure(
            datasetId = "owner/meta",
            configs = listOf(DatasetConfigInfo("default", listOf("train"), listOf("system", "url")))
        )
        assertEquals(null, AiDatasetResolver.resolve(structure, AiDatasetSuggestion(datasetId = "owner/meta")))
    }
}
