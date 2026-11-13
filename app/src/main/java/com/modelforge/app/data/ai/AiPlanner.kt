package com.modelforge.app.data.ai

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

@Serializable
data class AiDatasetSuggestion(
    val purpose: String = "general",
    @SerialName("dataset_id") val datasetId: String = "",
    val config: String = "",
    val split: String = "train",
    val columns: List<String> = emptyList(),
    val rows: Int = 1000,
    val weight: Float = 1f,
    val license: String? = null,
    @SerialName("commercial_use") val commercialUse: Boolean = false,
    val reason: String = ""
)

@Serializable
data class AiTrainingPlan(
    val steps: Int = 500,
    @SerialName("batch_size") val batchSize: Int = 8,
    @SerialName("seq_length") val seqLength: Int = 128,
    @SerialName("learning_rate") val learningRate: Float = 6e-4f,
    @SerialName("warmup_steps") val warmupSteps: Int = 25
)

@Serializable
data class AiPlan(
    val overview: String = "",
    val analogy: String = "",
    val training: AiTrainingPlan = AiTrainingPlan(),
    val datasets: List<AiDatasetSuggestion> = emptyList(),
    val notes: List<String> = emptyList(),
    val warnings: List<String> = emptyList()
)

sealed class AiAgentAction {
    data class Search(val query: String, val limit: Int) : AiAgentAction()
    data class Inspect(val datasetId: String, val config: String?) : AiAgentAction()
    data class Final(val plan: AiPlan) : AiAgentAction()
}

data class AiPlanRequest(
    val name: String,
    val creator: String,
    val architecture: String,
    val parameters: Long,
    val inputs: List<String>,
    val outputs: List<String>,
    val languages: List<String>,
    val trainingMinutes: Int,
    val context: Int,
    val speed: String,
    val reasoning: Boolean,
    val toolCalling: Boolean,
    val persona: String,
    val bits: Int,
    val currentDatasets: List<String>
)

object AiPlanner {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
        encodeDefaults = true
    }
    private val bareJson = Json { ignoreUnknownKeys = true; isLenient = true; coerceInputValues = true }

    fun systemPrompt(): String = """
        You are ModelForge's model architect. You design small, on-device models.
        Return exactly one JSON object and no markdown fences or commentary.
        Be honest: you cannot run training, so never claim measured results.
        Recommend real Hugging Face datasets only. Prefer datasets with clear,
        permissive, commercial-use licences; when the licence is unknown or
        non-commercial, say so in the licence field and warnings. Tamil must be
        included when it is selected. For every requested capability (general
        text, identity, reasoning, tool calls, each language, each modality)
        provide a dataset suggestion, but keep the whole plan compact: at most 8
        dataset suggestions total. Give rows and mix weights that fit a small
        phone model. Optimise for the most fluent, coherent and capable model the
        phone can train in training_minutes; never default to the smallest
        recipe. Prefer 3-6 complementary datasets with clean, long-form text,
        including a strong general corpus and instruction data. These are
        verified examples you may use when suitable:
        general English: HuggingFaceFW/fineweb-edu config sample-10BT, split
        train, column text; instruction: databricks/databricks-dolly-15k config
        default, split train, columns instruction,context,response; reasoning:
        openai/gsm8k config main, split train, columns question,answer; tool
        calls: glaiveai/glaive-function-calling-v2 config default, split train,
        columns system,chat; Tamil: mozhi-ai/tamil-corpus config default, split
        train, column text, or muthuramkumar/tamil_corpus_2.3m config default,
        split train, column text. If you are unsure about a dataset's config,
        use one of these verified examples instead. Use these architecture ids
        only:
        gpt_text (traditional transformer), ncm_text (NCM), ai_architecture
        (AI-designed experimental architecture). The app currently downloads
        text datasets only; for image, audio, video or documents, recommend data
        but warn that native training for it is not built yet.
        Choose training settings for quality, not speed. For models up to 10M
        parameters prefer 800-1600 steps, batch 8-12, sequence 128-256 and a
        learning rate of 8e-4 to 1e-3 when training_minutes allows. For 10-30M
        use 500-1000 steps, batch 4-8 and 4e-4 to 6e-4. For larger models use
        300-700 steps and 2e-4 to 4e-4. Use a warmup of 5-10% of steps. Dataset
        ids must be real, public Hugging Face repositories with text columns;
        do not invent ids. Return the final JSON promptly. Do not narrate or add
        extra sections.
        JSON shape:
        {
          "overview": "fluent plain-language overview with an analogy",
          "analogy": "one short analogy",
          "training": {"steps": 500, "batch_size": 8, "seq_length": 128,
                       "learning_rate": 0.0006, "warmup_steps": 25},
          "datasets": [
            {"purpose": "general|identity|reasoning|tool_calls|language|image|audio|video|documents",
             "dataset_id": "owner/name", "config": "", "split": "train",
             "columns": ["text"], "rows": 1000, "weight": 1.0,
             "license": "apache-2.0", "commercial_use": true,
             "reason": "why this data fits"}
          ],
          "notes": ["short design notes"],
          "warnings": ["honest limitations"]
        }
    """.trimIndent()

    fun userPrompt(request: AiPlanRequest): String = buildString {
        appendLine("Design this model:")
        appendLine("name: ${request.name}")
        appendLine("creator: ${request.creator}")
        appendLine("architecture: ${request.architecture}")
        appendLine("parameters: ${request.parameters}")
        appendLine("inputs: ${request.inputs.joinToString(", ")}")
        appendLine("outputs: ${request.outputs.joinToString(", ")}")
        appendLine("languages: ${request.languages.joinToString(", ")}")
        appendLine("training_minutes: ${request.trainingMinutes}")
        appendLine("context: ${request.context}")
        appendLine("speed: ${request.speed}")
        appendLine("reasoning: ${request.reasoning}")
        appendLine("tool_calling: ${request.toolCalling}")
        appendLine("compression_bits: ${request.bits}")
        appendLine("persona: ${request.persona.ifBlank { "(none)" }}")
        appendLine("current_datasets: ${request.currentDatasets.joinToString(", ").ifBlank { "(none)" }}")
        append("Choose the datasets, rows, mix weights and training settings.")
    }

    fun messages(request: AiPlanRequest): List<AiMessage> = listOf(
        AiMessage("system", systemPrompt()),
        AiMessage("user", userPrompt(request))
    )

    fun agentSystemPrompt(): String = """
        You are ModelForge's dataset research agent running in the background.
        You have live tools for the Hugging Face API. Reply with exactly one JSON
        object per turn and no markdown fences, XML, prose or commentary. Put the
        action name in a top-level string field named "action". Do not return
        multiple objects.
        Tools:
        1. Search real public datasets:
        {"action":"search","query":"keywords","limit":8}
        2. Inspect a dataset before using it:
        {"action":"inspect","dataset_id":"owner/name","config":"optional"}
        3. Finish only after verification:
        {"action":"final","plan":{...}}
        Rules:
        - Never invent dataset ids, configs, splits or columns. Use tool results.
        - Use search when you need candidates. Inspect every dataset you intend to
          use and copy its real config, split and text columns into the final plan.
        - If a requested config is missing, pick a real config from the inspection
          or search for another dataset. Do not guess.
        - If every inspected option is unusable, search again instead of repeating
          the same suggestion.
        - Prefer permissive, commercial-use datasets. If the licence is unknown or
          non-commercial, set commercial_use false and warn about it.
        - Choose 3-6 complementary datasets and quality-focused training settings.
        - Keep the final plan compact. The JSON plan shape is:
        {
          "overview": "plain-language overview with an analogy",
          "analogy": "one short analogy",
          "training": {"steps": 800, "batch_size": 10, "seq_length": 128,
                       "learning_rate": 0.0009, "warmup_steps": 80},
          "datasets": [
            {"purpose": "general|identity|reasoning|tool_calls|language",
             "dataset_id": "owner/name", "config": "real-config",
             "split": "real-split", "columns": ["real", "columns"],
             "rows": 1200, "weight": 1.0, "license": "real-license",
             "commercial_use": true, "reason": "why this data fits"}
          ],
          "notes": ["short design notes"],
          "warnings": ["honest limitations"]
        }
        ${systemPrompt()}
    """.trimIndent()

    fun agentMessages(request: AiPlanRequest): List<AiMessage> = listOf(
        AiMessage("system", agentSystemPrompt()),
        AiMessage(
            "user",
            userPrompt(request) + "\nUse the search and inspect tools. Do not finalize " +
                "until every dataset in the plan has been verified by a tool result. " +
                "Then return the final action."
        )
    )

    fun parseAgentAction(raw: String): AiAgentAction? {
        val text = raw.trim()
        if (text.isEmpty()) return null
        val candidates = buildList {
            add(text)
            val fenced = Regex("```(?:json)?\\s*(.*?)```", RegexOption.DOT_MATCHES_ALL)
                .findAll(text)
                .mapNotNull { it.groupValues.getOrNull(1)?.trim()?.takeIf { body -> body.isNotEmpty() } }
            addAll(fenced)
            addAll(extractJsonValues(text))
            val first = text.indexOf('{')
            val last = text.lastIndexOf('}')
            if (first >= 0 && last > first) add(text.substring(first, last + 1))
        }.distinct()
        for (candidate in candidates) {
            val element = runCatching { bareJson.parseToJsonElement(candidate) }.getOrNull() ?: continue
            parseActionElement(element)?.let { return it }
        }
        return null
    }

    private fun parseActionElement(element: JsonElement): AiAgentAction? {
        val obj = when (element) {
            is JsonObject -> element
            is JsonArray -> element.firstNotNullOfOrNull { it as? JsonObject } ?: return null
            else -> return null
        }

        val calls = obj["tool_calls"] as? JsonArray
        if (calls != null) {
            for (call in calls) {
                val entry = call as? JsonObject ?: continue
                val function = entry["function"] as? JsonObject
                val name = (function?.get("name") ?: entry["name"])?.asString() ?: continue
                val args = parseArgs(
                    function?.get("arguments") ?: entry["arguments"] ?:
                        function?.get("args") ?: entry["args"] ?: function
                )
                parseNamedAction(name, args)?.let { return it }
            }
        }

        val topName = obj["name"]?.asString()
        val topArgs = obj["arguments"] ?: obj["args"] ?: obj["parameters"]
        if (topName != null && topArgs != null) {
            parseNamedAction(topName, parseArgs(topArgs))?.let { return it }
        }

        val actionElement = obj["action"] ?: obj["tool"] ?: obj["type"]
        val actionName = when (actionElement) {
            is JsonPrimitive -> actionElement.contentOrNull
            is JsonObject -> (actionElement["type"] ?: actionElement["name"] ?: actionElement["action"])?.asString()
            else -> null
        }
        val actionArgs = when (actionElement) {
            is JsonObject -> actionElement
            else -> obj
        }
        if (actionName != null) {
            parseNamedAction(actionName, actionArgs)?.let { return it }
        }

        val planElement = obj["plan"]
        val plan = when (planElement) {
            is JsonObject -> decodePlan(planElement)
            is JsonPrimitive -> parseObject(planElement.contentOrNull)?.let { decodePlan(it) }
            else -> null
        }
        if (plan != null) return AiAgentAction.Final(plan)

        if (obj.containsKey("datasets") || obj.containsKey("training") ||
            obj.containsKey("overview") || obj.containsKey("analogy")
        ) {
            decodePlan(obj)?.let { return AiAgentAction.Final(it) }
        }
        return null
    }

    private fun parseNamedAction(rawName: String, args: JsonObject?): AiAgentAction? {
        val name = rawName.trim().lowercase().replace('-', '_').replace(' ', '_')
        val data = args ?: JsonObject(emptyMap())
        return when (name) {
            "search", "search_datasets", "dataset_search", "find_datasets", "lookup" -> {
                val query = stringField(data, "query", "q", "search", "term") ?: return null
                val limit = (stringField(data, "limit", "count", "max_results")
                    ?.toIntOrNull() ?: 8).coerceIn(1, 12)
                AiAgentAction.Search(query, limit)
            }
            "inspect", "inspect_dataset", "dataset_info", "get_dataset", "describe_dataset" -> {
                val id = stringField(data, "dataset_id", "datasetId", "id", "dataset", "repo_id")
                    ?: return null
                AiAgentAction.Inspect(id, stringField(data, "config", "config_name"))
            }
            "final", "finish", "final_answer", "done", "plan" -> {
                val planElement = data["plan"]
                val plan = when (planElement) {
                    is JsonObject -> decodePlan(planElement)
                    is JsonPrimitive -> parseObject(planElement.contentOrNull)?.let { decodePlan(it) }
                    else -> null
                } ?: decodePlan(data)
                plan?.let { AiAgentAction.Final(it) }
            }
            else -> null
        }
    }

    private fun parseArgs(element: JsonElement?): JsonObject? = when (element) {
        is JsonObject -> element
        is JsonPrimitive -> parseObject(element.contentOrNull)
        else -> null
    }

    private fun parseObject(text: String?): JsonObject? {
        if (text.isNullOrBlank()) return null
        val direct = runCatching { bareJson.parseToJsonElement(text) as? JsonObject }.getOrNull()
        if (direct != null) return direct
        return extractJsonValues(text).firstNotNullOfOrNull { candidate ->
            runCatching { bareJson.parseToJsonElement(candidate) as? JsonObject }.getOrNull()
        }
    }

    private fun JsonElement.asString(): String? = (this as? JsonPrimitive)?.contentOrNull?.trim()
        ?.takeIf { it.isNotEmpty() }

    private fun stringField(obj: JsonObject, vararg keys: String): String? {
        for (key in keys) {
            obj[key]?.asString()?.let { return it }
        }
        return null
    }

    private fun extractJsonValues(text: String): List<String> {
        val out = mutableListOf<String>()
        var index = 0
        while (index < text.length) {
            val open = text[index]
            if (open != '{' && open != '[') {
                index++
                continue
            }
            var depth = 0
            var inString = false
            var escaped = false
            var end = -1
            for (cursor in index until text.length) {
                val ch = text[cursor]
                if (inString) {
                    when {
                        escaped -> escaped = false
                        ch == '\\' -> escaped = true
                        ch == '"' -> inString = false
                    }
                } else {
                    when (ch) {
                        '"' -> inString = true
                        '{', '[' -> depth++
                        '}', ']' -> {
                            depth--
                            if (depth == 0) {
                                end = cursor
                                break
                            }
                        }
                    }
                }
            }
            if (end > index) {
                out += text.substring(index, end + 1)
                index = end + 1
            } else {
                index++
            }
        }
        return out
    }

    private fun decodePlan(obj: JsonObject): AiPlan? =
        runCatching { bareJson.decodeFromString<AiPlan>(obj.toString()) }.getOrNull()

    fun parse(raw: String): AiPlan {
        val text = raw.trim()
        if (text.isEmpty()) throw AiException("The AI returned an empty plan.")
        val candidates = buildList {
            add(text)
            val fenced = Regex("```(?:json)?\\s*(.*?)```", RegexOption.DOT_MATCHES_ALL)
                .find(text)?.groupValues?.getOrNull(1)?.trim()
            if (fenced != null) add(fenced)
            val first = text.indexOf('{')
            val last = text.lastIndexOf('}')
            if (first >= 0 && last > first) add(text.substring(first, last + 1))
        }
        var plan: AiPlan? = null
        var lastError: Throwable? = null
        for (candidate in candidates.distinct()) {
            val parsed = runCatching { bareJson.decodeFromString<AiPlan>(candidate) }.getOrElse {
                lastError = it
                null
            }
            if (parsed != null) {
                plan = parsed
                break
            }
        }
        val ok = plan ?: throw AiException(
            "The AI returned a plan in an unreadable format. ${lastError?.message.orEmpty()}".trim()
        )
        return sanitize(ok)
    }

    fun sanitize(plan: AiPlan): AiPlan {
        val training = plan.training.copy(
            steps = plan.training.steps.coerceIn(100, 2000),
            batchSize = plan.training.batchSize.coerceIn(1, 32),
            seqLength = plan.training.seqLength.coerceIn(32, 1024),
            learningRate = plan.training.learningRate.coerceIn(1e-5f, 1e-2f),
            warmupSteps = plan.training.warmupSteps.coerceIn(1, 400)
        )
        val seen = HashSet<String>()
        val datasets = plan.datasets.mapNotNull { item ->
            val id = item.datasetId.trim()
            val columns = item.columns.map { it.trim() }.filter { it.isNotBlank() }
            val config = item.config.trim().ifBlank { "default" }
            val split = item.split.trim().ifBlank { "train" }
            val key = "$id|$config|$split"
            if (id.isBlank() || !seen.add(key)) return@mapNotNull null
            item.copy(
                datasetId = id,
                purpose = item.purpose.trim().ifBlank { "general" },
                config = config,
                split = split,
                columns = columns,
                rows = item.rows.coerceIn(100, 5000),
                weight = item.weight.coerceIn(0.1f, 5f),
                license = item.license?.trim()?.takeIf { it.isNotBlank() },
                reason = item.reason.trim()
            )
        }
        val warnings = plan.warnings.toMutableList()
        datasets.filterNot { it.commercialUse }.forEach {
            warnings += "${it.datasetId}: commercial use is not confirmed. Check its licence."
        }
        return plan.copy(
            overview = plan.overview.trim(),
            analogy = plan.analogy.trim(),
            training = training,
            datasets = datasets,
            notes = plan.notes.map { it.trim() }.filter { it.isNotBlank() },
            warnings = warnings.distinct()
        )
    }

    fun toJson(plan: AiPlan): String = json.encodeToString(plan)
}
