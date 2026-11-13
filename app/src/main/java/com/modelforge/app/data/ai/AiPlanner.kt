package com.modelforge.app.data.ai

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

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
            val key = "$id|${item.config.trim()}|${item.split.trim()}"
            if (id.isBlank() || !seen.add(key)) return@mapNotNull null
            item.copy(
                datasetId = id,
                purpose = item.purpose.trim().ifBlank { "general" },
                config = item.config.trim(),
                split = item.split.trim().ifBlank { "train" },
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
