package com.modelforge.app.data.ai

import com.modelforge.app.data.hf.DatasetStructure

data class ResolvedAiDataset(
    val config: String,
    val split: String,
    val columns: List<String>
)

private val contentWords = listOf(
    "text", "content", "story", "article", "passage", "sentence", "document",
    "instruction", "response", "answer", "question", "solution", "prompt",
    "completion", "chat", "conversation", "input", "output", "translation"
)

private val lowValueColumns = setOf(
    "system", "id", "url", "source", "timestamp", "date", "license",
    "category", "language", "score", "file_path", "dump"
)

object AiDatasetResolver {
    fun resolve(structure: DatasetStructure, suggestion: AiDatasetSuggestion): ResolvedAiDataset? {
        val requestedConfig = suggestion.config.trim()
        val config = if (requestedConfig.isNotBlank()) {
            structure.configs.firstOrNull { it.name == requestedConfig && it.textColumns.isNotEmpty() }
        } else {
            structure.configs.firstOrNull { it.textColumns.isNotEmpty() }
                ?: structure.configs.firstOrNull()
        } ?: return null
        if (config.textColumns.isEmpty()) return null
        val split = config.splits.firstOrNull { it == suggestion.split }
            ?: config.splits.firstOrNull()
            ?: return null
        val wanted = suggestion.columns.filter { it in config.textColumns }
        val contentColumns = config.textColumns.filter { column ->
            val name = column.lowercase()
            contentWords.any { name.contains(it) }
        }
        val columns = wanted.ifEmpty { contentColumns.ifEmpty { config.textColumns } }.distinct()
        if (columns.all { it.lowercase() in lowValueColumns }) return null
        return ResolvedAiDataset(
            config = config.name,
            split = split,
            columns = columns
        )
    }
}
