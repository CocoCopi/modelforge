package com.modelforge.app.data.ai

import com.modelforge.app.data.hf.AiDatasetSource
import com.modelforge.app.data.hf.DatasetStructure
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

class AiDatasetAgent(
    private val chat: AiChat,
    private val source: AiDatasetSource,
    private val maxCalls: Int = 6,
    private val totalTimeoutMs: Long = 420_000L
) {
    suspend fun design(
        apiKey: String,
        request: AiPlanRequest,
        onStage: (String) -> Unit = {},
        onDelta: (String) -> Unit = {},
        onActivity: () -> Unit = {}
    ): AiPlan {
        val messages = AiPlanner.agentMessages(request).toMutableList()
        val structures = mutableMapOf<String, DatasetStructure>()
        var accepted: AiPlan? = null
        var lastError = "The AI did not finish a verified dataset plan."

        val completed = withTimeoutOrNull(totalTimeoutMs) {
            for (call in 0 until maxCalls) {
                onStage(if (call == 0) "Researching datasets with Hugging Face" else "AI is choosing the next step")
                val raw = chat.streamChat(apiKey, messages, onDelta, onActivity)
                val action = AiPlanner.parseAgentAction(raw)
                if (action == null) {
                    lastError = "The AI sent an unreadable tool action."
                    messages += AiMessage("assistant", raw.take(6000))
                    messages += AiMessage("user", invalidAction)
                    continue
                }
                when (action) {
                    is AiAgentAction.Search -> {
                        onStage("Searching Hugging Face for \"${action.query}\"")
                        val results = runCatching { source.search(action.query, action.limit) }.getOrNull()
                        messages += AiMessage("assistant", raw.take(6000))
                        messages += AiMessage(
                            "user",
                            "TOOL RESULT search:\n${searchResult(action.query, results)}"
                        )
                    }
                    is AiAgentAction.Inspect -> {
                        onStage("Inspecting ${action.datasetId}")
                        messages += AiMessage("assistant", raw.take(6000))
                        messages += AiMessage(
                            "user",
                            "TOOL RESULT inspect:\n${inspect(action.datasetId, action.config, structures)}"
                        )
                    }
                    is AiAgentAction.Final -> {
                        onStage("Verifying the AI's dataset choices")
                        val plan = verify(action.plan, structures)
                        if (plan.datasets.isEmpty() && action.plan.datasets.isNotEmpty()) {
                            val detail = plan.warnings.joinToString(" ").ifBlank {
                                "No dataset could be verified."
                            }
                            lastError = detail
                            messages += AiMessage("assistant", raw.take(6000))
                            messages += AiMessage(
                                "user",
                                "VERIFICATION FAILED: $detail\nUse search and inspect again, then " +
                                    "return a final action with only verified datasets."
                            )
                            continue
                        }
                        accepted = plan
                        return@withTimeoutOrNull true
                    }
                }
            }
            true
        }

        return accepted ?: throw AiException(
            if (completed == null) {
                "AI dataset research timed out. Check the connection and retry."
            } else {
                "The AI could not produce a verified dataset plan. $lastError"
            }
        )
    }

    private fun searchResult(query: String, results: List<com.modelforge.app.data.hf.HfDatasetSummary>?): String {
        return buildJsonObject {
            put("tool", "search")
            put("query", query)
            if (results == null) {
                put("error", "Hugging Face search failed.")
            } else {
                put("results", buildJsonArray {
                    results.forEach { result ->
                        add(buildJsonObject {
                            put("dataset_id", result.id)
                            put("downloads", JsonPrimitive(result.downloads))
                            put("likes", JsonPrimitive(result.likes))
                            put("license", result.resolvedLicense() ?: "unknown")
                        })
                    }
                })
            }
        }.toString().take(12_000)
    }

    private suspend fun inspect(
        datasetId: String,
        config: String?,
        structures: MutableMap<String, DatasetStructure>
    ): String {
        val detail = runCatching { source.detail(datasetId) }.getOrNull()
        val structure = runCatching { source.structure(datasetId) }.getOrNull()
        if (structure == null) {
            return buildJsonObject {
                put("tool", "inspect")
                put("dataset_id", datasetId)
                put("error", "Dataset metadata could not be read.")
            }.toString()
        }
        structures[datasetId] = structure
        val requested = config?.let { wanted -> structure.configs.firstOrNull { it.name == wanted } }
        val available = structure.configs.filter { it.textColumns.isNotEmpty() }
        val shown = when {
            requested != null -> listOf(requested)
            available.isNotEmpty() -> available.take(10)
            else -> structure.configs.take(10)
        }
        return buildJsonObject {
            put("tool", "inspect")
            put("dataset_id", datasetId)
            put("license", detail?.resolvedLicense() ?: "unknown")
            if (config != null && requested == null) {
                put("error", "Requested config '$config' does not exist. Use a real config from this result.")
            }
            put("configs", buildJsonArray {
                shown.forEach { entry ->
                    add(buildJsonObject {
                        put("name", entry.name)
                        put("splits", buildJsonArray { entry.splits.forEach { add(JsonPrimitive(it)) } })
                        put("text_columns", buildJsonArray { entry.textColumns.forEach { add(JsonPrimitive(it)) } })
                    })
                }
            })
            if (structure.configs.size > shown.size) {
                put("more_configs", JsonPrimitive(structure.configs.size))
            }
        }.toString().take(12_000)
    }

    private suspend fun verify(
        plan: AiPlan,
        structures: MutableMap<String, DatasetStructure>
    ): AiPlan {
        val dropped = mutableListOf<String>()
        val datasets = plan.datasets.mapNotNull { suggestion ->
            val id = suggestion.datasetId.trim()
            if (id.isBlank()) return@mapNotNull null
            val structure = structures[id] ?: runCatching { source.structure(id) }.getOrNull()
            if (structure == null) {
                dropped += "$id: metadata unavailable"
                return@mapNotNull null
            }
            structures[id] = structure
            val resolved = AiDatasetResolver.resolve(structure, suggestion)
            if (resolved == null) {
                dropped += "$id: no usable text config, split or columns"
                return@mapNotNull null
            }
            suggestion.copy(
                datasetId = id,
                config = resolved.config,
                split = resolved.split,
                columns = resolved.columns
            )
        }
        val warnings = plan.warnings + dropped.map { "$it. Skipped." }
        return AiPlanner.sanitize(plan.copy(datasets = datasets, warnings = warnings))
    }

    private companion object {
        const val invalidAction = "That was not a valid action. Reply with exactly one JSON object " +
            "using action search, inspect or final. Do not use markdown."
    }
}
