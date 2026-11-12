package com.modelforge.app.data.ai

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
