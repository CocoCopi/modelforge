package com.modelforge.app.data.ai

object AiDatasetFallbacks {
    fun forRequest(
        languages: List<String>,
        toolCalling: Boolean,
        reasoning: Boolean
    ): List<AiDatasetSuggestion> {
        val out = mutableListOf<AiDatasetSuggestion>()
        if (languages.any { it.contains("tamil", ignoreCase = true) }) {
            out += AiDatasetSuggestion(
                purpose = "language",
                datasetId = "mozhi-ai/tamil-corpus",
                config = "default",
                split = "train",
                columns = listOf("text"),
                rows = 1500,
                weight = 1f,
                license = "cc-by-sa-4.0",
                commercialUse = true,
                reason = "Verified Tamil text fallback."
            )
            out += AiDatasetSuggestion(
                purpose = "language",
                datasetId = "muthuramkumar/tamil_corpus_2.3m",
                config = "default",
                split = "train",
                columns = listOf("text"),
                rows = 1200,
                weight = 0.8f,
                license = "cc-by-sa-4.0",
                commercialUse = true,
                reason = "Second verified Tamil text fallback."
            )
        }
        if (toolCalling) {
            out += AiDatasetSuggestion(
                purpose = "tool_calls",
                datasetId = "glaiveai/glaive-function-calling-v2",
                config = "default",
                split = "train",
                columns = listOf("chat"),
                rows = 1000,
                weight = 0.8f,
                license = "apache-2.0",
                commercialUse = true,
                reason = "Verified function-calling fallback."
            )
        }
        if (reasoning) {
            out += AiDatasetSuggestion(
                purpose = "reasoning",
                datasetId = "openai/gsm8k",
                config = "main",
                split = "train",
                columns = listOf("question", "answer"),
                rows = 1000,
                weight = 0.8f,
                license = "mit",
                commercialUse = true,
                reason = "Verified short reasoning fallback."
            )
        }
        out += AiDatasetSuggestion(
            purpose = "instruction",
            datasetId = "databricks/databricks-dolly-15k",
            config = "default",
            split = "train",
            columns = listOf("instruction", "response"),
            rows = 1200,
            weight = 1f,
            license = "cc-by-sa-3.0",
            commercialUse = true,
            reason = "Verified instruction-following fallback."
        )
        out += AiDatasetSuggestion(
            purpose = "general",
            datasetId = "HuggingFaceFW/fineweb-edu",
            config = "sample-10BT",
            split = "train",
            columns = listOf("text"),
            rows = 1500,
            weight = 0.8f,
            license = "odc-by",
            commercialUse = true,
            reason = "Verified general English text fallback."
        )
        return out.distinctBy { "${it.datasetId}|${it.config}|${it.split}" }
    }
}
