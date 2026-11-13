package com.modelforge.app.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.modelforge.app.core.ArchitectureRegistry
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.core.IdentityCheck
import com.modelforge.app.core.ModelConfig
import com.modelforge.app.core.Modality
import com.modelforge.app.core.ResponseModes
import com.modelforge.app.data.DatasetMixEntry
import com.modelforge.app.data.ModelCard
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.ModelPreset
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.PresetStore
import com.modelforge.app.data.SecureTokenStore
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingSettings
import com.modelforge.app.data.WizardDraft
import com.modelforge.app.data.WizardDraftStore
import com.modelforge.app.data.ai.AiDatasetAgent
import com.modelforge.app.data.ai.AiDatasetFallbacks
import com.modelforge.app.data.ai.AiDatasetResolver
import com.modelforge.app.data.ai.AiDatasetSuggestion
import com.modelforge.app.data.ai.AiPlan
import com.modelforge.app.data.ai.AiPlanRequest
import com.modelforge.app.data.ai.AiTrainingPlan
import com.modelforge.app.data.ai.NvidiaClient
import com.modelforge.app.data.hf.DatasetStructure
import com.modelforge.app.data.hf.DownloadStats
import com.modelforge.app.data.hf.HfClient
import com.modelforge.app.data.hf.HfDatasetSummary
import com.modelforge.app.data.hf.HfException
import com.modelforge.app.data.hf.HfRepository
import com.modelforge.app.data.hf.isLikelyNonCommercial
import com.modelforge.app.data.tokenizer.BpeTokenizer
import com.modelforge.app.nativebridge.NativeModel
import com.modelforge.app.nativebridge.PackedModel
import com.modelforge.app.training.TrainingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

sealed class Screen {
    object Home : Screen()
    object Wizard : Screen()
    object Gallery : Screen()
    data class Training(val modelId: String) : Screen()
    data class ModelDetail(val modelId: String) : Screen()
    data class Playground(val modelId: String) : Screen()
}

data class WizardState(
    val modelId: String? = null,
    val name: String = "My first model",
    val creator: String = "",
    val architectureId: String = ArchitectureRegistry.GPT_TEXT,
    val targetParams: Long = 5_000_000L,
    val inputModalities: Set<Modality> = setOf(Modality.TEXT),
    val outputModalities: Set<Modality> = setOf(Modality.TEXT),
    val trainingMinutes: Int = 60,
    val contextChoice: Int = 0,
    val languages: Set<String> = setOf("English"),
    val speedMode: String = "normal",
    val reasoningEnabled: Boolean = false,
    val reasoningRoom: Int = 256,
    val toolCalling: Boolean = false,
    val persona: String = "",
    val config: ModelConfig? = null,
    val step: Int = 0,
    val bits: Int = 4,
    val steps: Int = 500,
    val batchSize: Int = 8,
    val seqLength: Int = 128,
    val learningRate: Float = 6e-4f,
    val warmupSteps: Int = 25,
    val checkpointEvery: Int = 50,
    val sampleEvery: Int = 100,
    val corpusChars: Int = 1_500_000,
    val searchQuery: String = "TinyStories",
    val searchResults: List<HfDatasetSummary> = emptyList(),
    val searching: Boolean = false,
    val selectedDataset: HfDatasetSummary? = null,
    val structure: DatasetStructure? = null,
    val configName: String = "",
    val splitName: String = "",
    val columns: Set<String> = emptySet(),
    val rows: Int = 1000,
    val weight: Float = 1f,
    val datasetBusy: Boolean = false,
    val datasetProgress: DownloadStats? = null,
    val datasets: List<DatasetMixEntry> = emptyList(),
    val error: String? = null
) {
    val canStart: Boolean
        get() = modelId != null && config != null && name.isNotBlank() &&
            creator.isNotBlank() && datasets.isNotEmpty() &&
            ArchitectureRegistry.info(architectureId).trainable &&
            inputModalities == setOf(Modality.TEXT) &&
            outputModalities == setOf(Modality.TEXT)
}

data class AiPlannerState(
    val busy: Boolean = false,
    val streaming: String = "",
    val plan: AiPlan? = null,
    val applied: Boolean = false,
    val stage: String = "",
    val appliedDatasetIds: Set<String> = emptySet(),
    val skippedDatasetIds: Set<String> = emptySet(),
    val skippedDatasetReasons: Map<String, String> = emptyMap(),
    val startedAt: Long = 0L,
    val events: Int = 0,
    val error: String? = null
)

data class PlaygroundState(
    val modelId: String = "",
    val prompt: String = "Once upon a time",
    val maxNew: Int = 120,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val mode: String = "checkpoint",
    val bits: Int = 4,
    val speedMode: String = ResponseModes.NORMAL,
    val reasoningEnabled: Boolean = false,
    val thinkingExpanded: Boolean = false,
    val output: String = "",
    val busy: Boolean = false,
    val status: String = "",
    val tokensPerSec: Double = 0.0,
    val latencyMs: Long = 0L,
    val generatedTokens: Int = 0,
    val exportedBytes: Long = 0L,
    val error: String? = null
) {
    companion object {
        const val MODE_CHECKPOINT = "checkpoint"
        const val MODE_PACKED = "packed"
    }
}

data class IdentityAnswer(
    val question: String,
    val response: String,
    val passed: Boolean
)

data class IdentityCheckState(
    val modelId: String = "",
    val busy: Boolean = false,
    val answers: List<IdentityAnswer> = emptyList(),
    val error: String? = null
) {
    val passed: Int get() = answers.count { it.passed }
    val total: Int get() = answers.size
    val score: Double get() = if (total <= 0) 0.0 else passed.toDouble() / total
    val completed: Boolean get() = !busy && total > 0 && error == null
    val needsReinforce: Boolean get() = completed && score < IdentityCheck.PASS_THRESHOLD
}

class ModelForgeViewModel(app: Application) : AndroidViewModel(app) {

    private val store = ModelStore(app)
    private val draftStore = WizardDraftStore(app)
    private val presetStore = PresetStore(app)
    private val tokenStore = SecureTokenStore(app)
    private val _aiPlanner = MutableStateFlow(AiPlannerState())
    val aiPlanner: StateFlow<AiPlannerState> = _aiPlanner.asStateFlow()
    private val hf = HfRepository(HfClient.create { tokenStore.token }) { message ->
        _aiPlanner.update { if (it.busy) it.copy(stage = message) else it }
    }
    private val nvidia = NvidiaClient()
    private val agent = AiDatasetAgent(nvidia, hf)
    private var aiJob: kotlinx.coroutines.Job? = null
    private var lastAiRequest: AiPlanRequest? = null
    private var aiRateLimited = false

    val hardware = HardwareDetector.detect(app)
    val maxTargetParams: Long = hardware.maxTrainableParams

    private val _screen = MutableStateFlow<Screen>(Screen.Home)
    val screen: StateFlow<Screen> = _screen.asStateFlow()

    private val _models = MutableStateFlow(store.listManifests())
    val models: StateFlow<List<ModelManifest>> = _models.asStateFlow()

    private val _wizard = MutableStateFlow(WizardState())
    val wizard: StateFlow<WizardState> = _wizard.asStateFlow()

    private val _presets = MutableStateFlow(presetStore.list())
    val presets: StateFlow<List<ModelPreset>> = _presets.asStateFlow()

    private val _tokenPresent = MutableStateFlow(tokenStore.token != null)
    val tokenPresent: StateFlow<Boolean> = _tokenPresent.asStateFlow()

    private val _aiKeyPresent = MutableStateFlow(tokenStore.nvidiaToken != null)
    val aiKeyPresent: StateFlow<Boolean> = _aiKeyPresent.asStateFlow()

    private val _notice = MutableStateFlow<String?>(null)
    val notice: StateFlow<String?> = _notice.asStateFlow()

    private fun updateWizard(transform: (WizardState) -> WizardState) {
        _wizard.value = transform(_wizard.value).also(::persistDraft)
    }

    private fun persistDraft(state: WizardState) {
        val id = state.modelId ?: return
        draftStore.save(
            WizardDraft(
                modelId = id,
                name = state.name,
                creator = state.creator,
                architectureId = state.architectureId,
                targetParams = state.targetParams,
                inputModalities = state.inputModalities.toList(),
                outputModalities = state.outputModalities.toList(),
                trainingMinutes = state.trainingMinutes,
                contextLength = state.contextChoice,
                languages = state.languages.toList(),
                speedMode = state.speedMode,
                reasoningEnabled = state.reasoningEnabled,
                reasoningRoom = state.reasoningRoom,
                toolCalling = state.toolCalling,
                persona = state.persona,
                bits = state.bits,
                steps = state.steps,
                batchSize = state.batchSize,
                seqLength = state.seqLength,
                learningRate = state.learningRate,
                warmupSteps = state.warmupSteps,
                checkpointEvery = state.checkpointEvery,
                sampleEvery = state.sampleEvery,
                corpusChars = state.corpusChars,
                datasets = state.datasets,
                step = state.step
            )
        )
    }

    private val _playground = MutableStateFlow(PlaygroundState())
    val playground: StateFlow<PlaygroundState> = _playground.asStateFlow()

    private val _identityCheck = MutableStateFlow(IdentityCheckState())
    val identityCheck: StateFlow<IdentityCheckState> = _identityCheck.asStateFlow()

    fun openGallery() { _screen.value = Screen.Gallery }

    fun manifest(id: String): ModelManifest? = store.loadManifest(id)

    fun liveState(id: String): TrainingLiveState? = store.loadLiveState(id)

    fun checkpointMegabytes(id: String): Pair<Long, Long> {
        fun mb(file: File): Long = if (file.exists()) file.length() / (1024L * 1024L) else 0L
        return mb(store.lastCheckpoint(id)) to mb(store.bestCheckpoint(id))
    }

    fun targetOptions(): List<Long> {
        val cap = maxTargetParams
        val base = listOf(1_000_000L, 2_000_000L, 5_000_000L, 10_000_000L, 25_000_000L, 50_000_000L)
            .filter { it <= cap }
            .toMutableList()
        if (base.isEmpty()) base.add(cap)
        if (cap > base.last()) base.add(cap)
        return base.distinct().sorted()
    }

    fun startNewModel() {
        val draft = draftStore.load()
        val target = draft?.targetParams ?: defaultTarget()
        val architectureId = draft?.architectureId ?: ArchitectureRegistry.GPT_TEXT
        val context = draft?.contextLength?.takeIf { it > 0 }
        val config = ArchitectureRegistry.derive(
            architectureId,
            target,
            maxTargetParams,
            context
        ).config
        _wizard.value = if (draft != null && config != null) {
            WizardState(
                modelId = draft.modelId.ifBlank { store.newModelId("model") },
                name = draft.name,
                creator = draft.creator,
                architectureId = architectureId,
                targetParams = target,
                inputModalities = (draft.inputModalities.ifEmpty { listOf(Modality.TEXT) }).toSet(),
                outputModalities = (draft.outputModalities.ifEmpty { listOf(Modality.TEXT) }).toSet(),
                trainingMinutes = draft.trainingMinutes.coerceIn(15, 480),
                contextChoice = draft.contextLength,
                languages = draft.languages.toSet(),
                speedMode = draft.speedMode,
                reasoningEnabled = draft.reasoningEnabled,
                reasoningRoom = draft.reasoningRoom,
                toolCalling = draft.toolCalling,
                persona = draft.persona,
                bits = draft.bits,
                steps = draft.steps,
                batchSize = draft.batchSize,
                seqLength = draft.seqLength,
                learningRate = draft.learningRate,
                warmupSteps = draft.warmupSteps,
                checkpointEvery = draft.checkpointEvery,
                sampleEvery = draft.sampleEvery,
                corpusChars = draft.corpusChars,
                datasets = draft.datasets,
                step = draft.step.coerceIn(0, WizardStep.lastIndex),
                config = config
            )
        } else {
            val fresh = ConfigDeriver.derive(
                target,
                ConfigDeriver.Options(maxParams = maxTargetParams)
            ).config.copy(architectureId = ArchitectureRegistry.GPT_TEXT)
            WizardState(
                modelId = store.newModelId("model"),
                targetParams = target
            ).withSettings(fresh)
        }
        _notice.value = null
        _aiPlanner.value = AiPlannerState()
        _screen.value = Screen.Wizard
    }

    fun updateName(value: String) {
        updateWizard { it.copy(name = value) }
    }

    fun updateCreator(value: String) {
        updateWizard { it.copy(creator = value) }
    }

    fun updateArchitecture(id: String) {
        val info = ArchitectureRegistry.info(id)
        if (!info.available) {
            updateWizard { it.copy(error = info.unavailableReason ?: "This architecture is not available.") }
            return
        }
        val state = _wizard.value
        val config = ArchitectureRegistry.derive(
            id,
            state.targetParams,
            maxTargetParams,
            state.contextChoice.takeIf { it > 0 }
        ).config
        updateWizard {
            it.copy(
                architectureId = id,
                config = config,
                inputModalities = setOf(Modality.TEXT),
                outputModalities = setOf(Modality.TEXT),
                bits = if (it.bits in info.supportedBits) it.bits else info.supportedBits.firstOrNull() ?: 4,
                error = null
            )
        }
    }

    fun updateTarget(target: Long) {
        val capped = target.coerceIn(1_000_000L, maxTargetParams)
        val state = _wizard.value
        val config = ArchitectureRegistry.derive(
            state.architectureId,
            capped,
            maxTargetParams,
            state.contextChoice.takeIf { it > 0 }
        ).config
        updateWizard {
            if (config == null) it.copy(targetParams = capped)
            else it.copy(targetParams = capped).withSettings(config)
        }
    }

    fun updateContext(length: Int) {
        val state = _wizard.value
        if (length <= 0) return
        val config = ArchitectureRegistry.derive(
            state.architectureId,
            state.targetParams,
            maxTargetParams,
            length
        ).config
        updateWizard {
            it.copy(contextChoice = length, config = config ?: it.config, error = null)
        }
    }

    fun toggleLanguage(language: String) {
        updateWizard { state ->
            val next = if (language in state.languages) state.languages - language else state.languages + language
            state.copy(languages = next, error = null)
        }
    }

    fun toggleInputModality(modality: Modality) {
        updateWizard { state ->
            val next = if (modality in state.inputModalities) state.inputModalities - modality
            else state.inputModalities + modality
            if (next.isEmpty()) state.copy(error = "Keep at least one input modality.") else
                state.copy(inputModalities = next, error = null)
        }
    }

    fun toggleOutputModality(modality: Modality) {
        updateWizard { state ->
            val next = if (modality in state.outputModalities) state.outputModalities - modality
            else state.outputModalities + modality
            if (next.isEmpty()) state.copy(error = "Keep at least one output modality.") else
                state.copy(outputModalities = next, error = null)
        }
    }

    fun updateTrainingMinutes(minutes: Int) {
        updateWizard { it.copy(trainingMinutes = minutes.coerceIn(15, 480), error = null) }
    }

    fun saveAiKey(value: String) {
        if (value.isNotBlank() && !tokenStore.isAvailable) {
            _aiPlanner.value = AiPlannerState(error = "Secure storage is unavailable on this device, so the key was not saved.")
            return
        }
        tokenStore.nvidiaToken = value
        _aiKeyPresent.value = tokenStore.nvidiaToken != null
        _aiPlanner.value = AiPlannerState(
            error = when {
                value.isBlank() -> "NVIDIA API key cleared."
                !_aiKeyPresent.value -> "The key could not be saved securely."
                else -> null
            }
        )
    }

    fun clearAiKey() {
        tokenStore.clearNvidia()
        _aiKeyPresent.value = false
        _aiPlanner.value = AiPlannerState(error = "NVIDIA API key cleared. AI planning is off.")
    }

    fun requestAiPlan() {
        val state = _wizard.value
        val key = tokenStore.nvidiaToken
        if (key.isNullOrBlank()) {
            _aiPlanner.value = AiPlannerState(error = "Add your NVIDIA API key first. It is stored encrypted and never logged.")
            return
        }
        if (_aiPlanner.value.busy) return
        val req = AiPlanRequest(
            name = state.name,
            creator = state.creator,
            architecture = state.architectureId,
            parameters = state.config?.paramCount() ?: state.targetParams,
            inputs = state.inputModalities.map { it.label },
            outputs = state.outputModalities.map { it.label },
            languages = state.languages.toList(),
            trainingMinutes = state.trainingMinutes,
            context = state.contextChoice.takeIf { it > 0 } ?: state.config?.contextLength ?: 128,
            speed = state.speedMode,
            reasoning = state.reasoningEnabled,
            toolCalling = state.toolCalling,
            persona = state.persona,
            bits = state.bits,
            currentDatasets = state.datasets.map { it.datasetId }
        )
        lastAiRequest = req
        _aiPlanner.value = AiPlannerState(
            busy = true,
            streaming = "",
            stage = "Contacting NVIDIA",
            startedAt = System.currentTimeMillis()
        )
        aiJob = viewModelScope.launch {
            try {
                val plan = agent.design(
                    key,
                    req,
                    onStage = { stage ->
                        _aiPlanner.update { it.copy(stage = stage) }
                    },
                    onDelta = { delta ->
                        _aiPlanner.update {
                            it.copy(streaming = (it.streaming + delta).takeLast(12000))
                        }
                    },
                    onActivity = {
                        _aiPlanner.update { it.copy(events = it.events + 1) }
                    }
                )
                _aiPlanner.value = AiPlannerState(
                    busy = true,
                    plan = plan,
                    stage = "Applying the AI training recipe",
                    startedAt = _aiPlanner.value.startedAt,
                    events = _aiPlanner.value.events
                )
                applyAiTraining(plan.training)
                autoApplyDatasets(plan, req)
                finishAiPlan(plan)
            } catch (t: kotlinx.coroutines.CancellationException) {
                throw t
            } catch (t: Throwable) {
                if (!isActive) return@launch
                _aiPlanner.update {
                    it.copy(
                        busy = false,
                        stage = "",
                        error = t.message ?: "AI planning failed. Check the key and connection."
                    )
                }
            } finally {
                aiJob = null
            }
        }
    }

    private fun finishAiPlan(plan: AiPlan) {
        val applied = _aiPlanner.value.appliedDatasetIds
        val skipped = _aiPlanner.value.skippedDatasetIds
        val datasetCount = _wizard.value.datasets.size
        if (datasetCount > 0) {
            updateWizard {
                if (it.step == WizardStep.AI_PLAN.ordinal) {
                    it.copy(step = WizardStep.REVIEW.ordinal, error = null)
                } else it.copy(error = null)
            }
            val fallbackApplied = applied - plan.datasets.map { it.datasetId }.toSet()
            _aiPlanner.update {
                it.copy(
                    busy = false,
                    applied = true,
                    stage = when {
                        applied.isEmpty() -> "Applied training settings; kept the existing dataset mix."
                        fallbackApplied.isNotEmpty() -> "AI suggestions were unavailable; downloaded " +
                            "${fallbackApplied.size} verified starter dataset(s)." +
                            if (skipped.isEmpty()) "" else " Skipped ${skipped.size} suggestion(s)."
                        else -> "Applied training settings and downloaded ${applied.size} dataset(s)." +
                            if (skipped.isEmpty()) "" else " Skipped ${skipped.size} unavailable suggestion(s)."
                    }
                )
            }
        } else {
            val reasons = _aiPlanner.value.skippedDatasetReasons.values
                .distinct()
                .take(2)
                .joinToString(" ")
            _aiPlanner.update {
                it.copy(
                    busy = false,
                    applied = false,
                    stage = "Training settings applied.",
                    error = when {
                        aiRateLimited -> "Hugging Face is rate-limiting this network, so dataset " +
                            "downloads stopped. Wait a minute, add a Hugging Face token on Home for " +
                            "higher limits, or switch network, then tap Retry dataset downloads. $reasons"
                        plan.datasets.isEmpty() -> "The AI returned no usable dataset suggestions, and " +
                            "the verified fallbacks could not be downloaded. Check your connection " +
                            "and retry. $reasons"
                        else -> "None of the AI's ${plan.datasets.size} dataset suggestion(s) could " +
                            "be downloaded, and the verified fallbacks failed too. $reasons"
                    }.trim()
                )
            }
        }
    }

    fun retryAiDatasets() {
        val plan = _aiPlanner.value.plan ?: return
        val req = lastAiRequest ?: return
        if (_aiPlanner.value.busy) return
        _aiPlanner.value = _aiPlanner.value.copy(
            busy = true,
            streaming = "",
            applied = false,
            stage = "Retrying dataset downloads",
            appliedDatasetIds = emptySet(),
            skippedDatasetIds = emptySet(),
            skippedDatasetReasons = emptyMap(),
            startedAt = System.currentTimeMillis(),
            error = null
        )
        aiJob = viewModelScope.launch {
            try {
                autoApplyDatasets(plan, req)
                finishAiPlan(plan)
            } catch (t: kotlinx.coroutines.CancellationException) {
                throw t
            } catch (t: Throwable) {
                if (!isActive) return@launch
                _aiPlanner.update {
                    it.copy(
                        busy = false,
                        stage = "",
                        error = t.message ?: "Dataset download failed. Check the connection and retry."
                    )
                }
            } finally {
                aiJob = null
            }
        }
    }

    fun cancelAiPlan() {
        aiJob?.cancel()
        aiJob = null
        nvidia.cancel()
        _aiPlanner.update {
            it.copy(busy = false, error = "AI request cancelled. You can retry when ready.")
        }
    }

    override fun onCleared() {
        aiJob?.cancel()
        nvidia.cancel()
        super.onCleared()
    }

    private fun applyAiTraining(training: AiTrainingPlan) {
        updateWizard { state ->
            val seq = training.seqLength.coerceAtMost(
                state.contextChoice.takeIf { it > 0 } ?: state.config?.contextLength ?: training.seqLength
            )
            val steps = training.steps.coerceIn(100, 2000)
            state.copy(
                steps = steps,
                batchSize = training.batchSize.coerceIn(1, 32),
                seqLength = seq.coerceAtLeast(32),
                learningRate = training.learningRate.coerceIn(1e-5f, 1e-2f),
                warmupSteps = training.warmupSteps.coerceIn(1, steps),
                checkpointEvery = maxOf(20, steps / 10),
                sampleEvery = maxOf(20, steps / 5),
                corpusChars = (steps.toLong() * training.batchSize * seq * 3L)
                    .coerceIn(200_000L, 2_000_000L)
                    .toInt(),
                error = null
            )
        }
    }

    private suspend fun autoApplyDatasets(plan: AiPlan, request: AiPlanRequest) {
        val modelId = _wizard.value.modelId ?: return
        val dataDir = store.dataDir(modelId)
        val before = _wizard.value.datasets.size
        aiRateLimited = false
        val suggestions = plan.datasets.take(6)
        for ((position, suggestion) in suggestions.withIndex()) {
            tryApplyDataset(suggestion, "${position + 1}/${suggestions.size}", dataDir)
            if (aiRateLimited) return
        }
        if (_wizard.value.datasets.size > before || _wizard.value.datasets.isNotEmpty()) return

        val fallbacks = AiDatasetFallbacks.forRequest(
            languages = request.languages,
            toolCalling = request.toolCalling,
            reasoning = request.reasoning
        ).filter { fallback -> suggestions.none { it.datasetId == fallback.datasetId } }
        var fallbacksApplied = 0
        for ((position, fallback) in fallbacks.withIndex()) {
            currentCoroutineContext().ensureActive()
            _aiPlanner.update { it.copy(stage = "Trying verified starter data ${position + 1}/${fallbacks.size}") }
            if (tryApplyDataset(fallback, "fallback ${position + 1}/${fallbacks.size}", dataDir)) {
                fallbacksApplied++
                if (fallbacksApplied >= 2) return
            }
            if (aiRateLimited) return
        }
    }

    private suspend fun tryApplyDataset(
        suggestion: AiDatasetSuggestion,
        label: String,
        dataDir: File
    ): Boolean {
        currentCoroutineContext().ensureActive()
        val id = suggestion.datasetId.trim()
        if (id.isBlank()) return false
        if (_wizard.value.datasets.any { it.datasetId == id }) {
            _aiPlanner.update {
                it.copy(
                    skippedDatasetIds = it.skippedDatasetIds + id,
                    skippedDatasetReasons = it.skippedDatasetReasons + (id to "Already in the dataset mix."),
                    stage = "Already in the mix: $id"
                )
            }
            return false
        }
        _aiPlanner.update { it.copy(stage = "Checking $id ($label)") }
        val destination = File(dataDir, "ai_${UUID.randomUUID()}.txt")
        return try {
            val detail = hf.detail(id)
            val structure = hf.structure(id)
            val resolved = AiDatasetResolver.resolve(structure, suggestion)
                ?: throw HfException("No usable text config, split or column was found.")
            val rows = suggestion.rows.coerceIn(100, 1000)
            val maxBytes = (rows.toLong() * 2048L).coerceIn(256_000L, 1_500_000L)
            _aiPlanner.update { it.copy(stage = "Downloading $id ($label)") }
            val stats = hf.downloadText(
                datasetId = id,
                config = resolved.config,
                split = resolved.split,
                columns = resolved.columns,
                maxRows = rows,
                maxBytes = maxBytes,
                destination = destination
            ) { progress ->
                _aiPlanner.update {
                    it.copy(stage = "Downloading $id · ${progress.rows} rows ($label)")
                }
            }
            currentCoroutineContext().ensureActive()
            val license = detail.resolvedLicense()
            val entry = DatasetMixEntry(
                datasetId = id,
                config = resolved.config,
                split = resolved.split,
                columns = resolved.columns,
                weight = suggestion.weight.coerceIn(0.1f, 5f),
                cacheFile = destination.name,
                rows = stats.rows,
                bytes = stats.bytes,
                license = license,
                nonCommercial = isLikelyNonCommercial(license)
            )
            updateWizard { it.copy(datasets = it.datasets + entry, error = null) }
            _aiPlanner.update {
                it.copy(
                    appliedDatasetIds = it.appliedDatasetIds + id,
                    skippedDatasetIds = it.skippedDatasetIds - id,
                    skippedDatasetReasons = it.skippedDatasetReasons - id,
                    stage = "Downloaded $id ($label)"
                )
            }
            true
        } catch (t: kotlinx.coroutines.CancellationException) {
            destination.delete()
            throw t
        } catch (t: Throwable) {
            destination.delete()
            val reason = (t.message ?: "unavailable").take(180)
            if (reason.contains("rate limit", ignoreCase = true)) aiRateLimited = true
            _aiPlanner.update {
                it.copy(
                    skippedDatasetIds = it.skippedDatasetIds + id,
                    skippedDatasetReasons = it.skippedDatasetReasons + (id to reason),
                    stage = "Skipped $id: $reason"
                )
            }
            false
        }
    }

    fun goToDatasets() {
        updateWizard { it.copy(step = WizardStep.DATASETS.ordinal) }
    }

    fun goToAiPlan() {
        updateWizard { it.copy(step = WizardStep.AI_PLAN.ordinal) }
    }

    fun updateSpeedMode(mode: String) {
        if (mode !in setOf("instant", "normal", "extra")) return
        updateWizard { it.copy(speedMode = mode) }
    }

    fun updateReasoning(enabled: Boolean) {
        updateWizard { it.copy(reasoningEnabled = enabled) }
    }

    fun updateReasoningRoom(tokens: Int) {
        updateWizard { it.copy(reasoningRoom = tokens.coerceIn(64, 2048)) }
    }

    fun updateToolCalling(enabled: Boolean) {
        updateWizard { it.copy(toolCalling = enabled) }
    }

    fun updatePersona(value: String) {
        updateWizard { it.copy(persona = value.take(600)) }
    }

    fun updateBits(bits: Int) {
        val supported = ArchitectureRegistry.supportedBits(_wizard.value.architectureId)
        if (bits !in supported) return
        updateWizard { it.copy(bits = bits) }
    }

    fun updateSteps(value: Int) {
        val steps = value.coerceIn(100, 2000)
        updateWizard {
            it.copy(
                steps = steps,
                warmupSteps = maxOf(10, steps / 20),
                checkpointEvery = maxOf(20, steps / 10),
                sampleEvery = maxOf(20, steps / 5),
                corpusChars = (steps.toLong() * it.batchSize * it.seqLength * 3L)
                    .coerceIn(200_000L, 2_000_000L)
                    .toInt()
            )
        }
    }

    fun savePreset(presetName: String) {
        val label = presetName.trim()
        if (label.isBlank()) return
        val state = _wizard.value
        val preset = ModelPreset(
            id = UUID.randomUUID().toString(),
            name = label.take(60),
            architectureId = state.architectureId,
            targetParams = state.targetParams,
            contextLength = state.contextChoice,
            languages = state.languages.toList(),
            speedMode = state.speedMode,
            reasoningEnabled = state.reasoningEnabled,
            reasoningRoom = state.reasoningRoom,
            toolCalling = state.toolCalling,
            persona = state.persona,
            bits = state.bits,
            steps = state.steps,
            batchSize = state.batchSize,
            seqLength = state.seqLength,
            learningRate = state.learningRate,
            warmupSteps = state.warmupSteps,
            checkpointEvery = state.checkpointEvery,
            sampleEvery = state.sampleEvery,
            corpusChars = state.corpusChars
        )
        _presets.value = presetStore.save(preset)
        _notice.value = "Preset \"${preset.name}\" saved. Datasets are not part of a preset."
    }

    fun applyPreset(id: String) {
        val preset = _presets.value.firstOrNull { it.id == id } ?: return
        val info = ArchitectureRegistry.info(preset.architectureId)
        if (!info.available) {
            _notice.value = info.unavailableReason ?: "That architecture is not available."
            return
        }
        val derive = ArchitectureRegistry.derive(
            preset.architectureId,
            preset.targetParams.coerceIn(1_000_000L, maxTargetParams),
            maxTargetParams,
            preset.contextLength.takeIf { it > 0 }
        )
        val config = derive.config
        if (config == null) {
            _notice.value = derive.explanation.ifBlank { "Could not derive that preset." }
            return
        }
        updateWizard {
            it.copy(
                architectureId = preset.architectureId,
                targetParams = preset.targetParams,
                contextChoice = preset.contextLength,
                languages = preset.languages.toSet(),
                speedMode = preset.speedMode,
                reasoningEnabled = preset.reasoningEnabled,
                reasoningRoom = preset.reasoningRoom,
                toolCalling = preset.toolCalling,
                persona = preset.persona,
                bits = if (preset.bits in info.supportedBits) preset.bits else info.supportedBits.firstOrNull() ?: 4,
                steps = preset.steps,
                batchSize = preset.batchSize,
                seqLength = preset.seqLength,
                learningRate = preset.learningRate,
                warmupSteps = preset.warmupSteps,
                checkpointEvery = preset.checkpointEvery,
                sampleEvery = preset.sampleEvery,
                corpusChars = preset.corpusChars,
                config = config,
                error = null
            )
        }
        _notice.value = "Preset \"${preset.name}\" applied. Add datasets before training."
    }

    fun deletePreset(id: String) {
        _presets.value = presetStore.delete(id)
    }

    fun nextStep() {
        updateWizard { it.copy(step = (it.step + 1).coerceAtMost(WizardStep.lastIndex), error = null) }
    }

    fun previousStep() {
        updateWizard { it.copy(step = (it.step - 1).coerceAtLeast(0), error = null) }
    }

    fun wizardBack() {
        if (_wizard.value.step > 0) {
            previousStep()
        } else {
            discardWizard()
        }
    }

    fun discardWizard() {
        draftStore.clear()
        val id = _wizard.value.modelId
        if (id != null && store.loadManifest(id) == null) {
            store.deleteModel(id)
        }
        _wizard.value = WizardState()
        _screen.value = Screen.Home
        refreshModels()
    }

    fun updateSearchQuery(value: String) {
        updateWizard { it.copy(searchQuery = value) }
    }

    fun searchDatasets() {
        val query = _wizard.value.searchQuery.trim()
        if (query.length < 2) {
            updateWizard { it.copy(error = "Type at least two characters to search.") }
            return
        }
        viewModelScope.launch {
            updateWizard { it.copy(searching = true, error = null) }
            try {
                val results = hf.search(query, limit = 25)
                updateWizard {
                    it.copy(
                        searchResults = results,
                        searching = false,
                        error = if (results.isEmpty()) "No datasets matched \"$query\"." else null
                    )
                }
            } catch (t: Throwable) {
                updateWizard {
                    it.copy(searching = false, error = t.message ?: "Dataset search failed.")
                }
            }
        }
    }

    fun selectDataset(summary: HfDatasetSummary) {
        updateWizard {
            it.copy(
                selectedDataset = summary,
                structure = null,
                configName = "",
                splitName = "",
                columns = emptySet(),
                datasetBusy = true,
                datasetProgress = null,
                error = null
            )
        }
        viewModelScope.launch {
            try {
                val structure = hf.structure(summary.id)
                val first = structure.configs.firstOrNull()
                updateWizard {
                    it.copy(
                        structure = structure,
                        configName = first?.name.orEmpty(),
                        splitName = first?.splits?.firstOrNull().orEmpty(),
                        columns = first?.textColumns?.toSet() ?: emptySet(),
                        datasetBusy = false,
                        error = if (structure.configs.isEmpty()) {
                            "No configs were found for ${summary.id}."
                        } else null
                    )
                }
            } catch (t: Throwable) {
                updateWizard {
                    it.copy(datasetBusy = false, error = t.message ?: "Could not inspect this dataset.")
                }
            }
        }
    }

    fun selectConfig(name: String) {
        updateWizard { state ->
            val cfg = state.structure?.configs?.firstOrNull { it.name == name }
            state.copy(
                configName = name,
                splitName = cfg?.splits?.firstOrNull().orEmpty(),
                columns = cfg?.textColumns?.toSet() ?: emptySet()
            )
        }
    }

    fun selectSplit(name: String) {
        updateWizard { it.copy(splitName = name) }
    }

    fun toggleColumn(name: String) {
        updateWizard { state ->
            val columns = if (name in state.columns) state.columns - name else state.columns + name
            state.copy(columns = columns)
        }
    }

    fun updateRows(value: Int) {
        updateWizard { it.copy(rows = value.coerceIn(100, 5000)) }
    }

    fun updateWeight(value: Float) {
        updateWizard { it.copy(weight = value.coerceIn(0.1f, 5f)) }
    }

    fun downloadSelectedDataset() {
        val state = _wizard.value
        val modelId = state.modelId ?: return
        val summary = state.selectedDataset ?: return
        if (state.configName.isBlank() || state.splitName.isBlank() || state.columns.isEmpty()) {
            updateWizard {
                it.copy(error = "Choose a config, split and at least one text column.")
            }
            return
        }
        val index = state.datasets.size
        val destination = File(store.dataDir(modelId), "dataset_$index.txt")
        viewModelScope.launch {
            updateWizard { it.copy(datasetBusy = true, datasetProgress = null, error = null) }
            try {
                val stats = hf.downloadText(
                    datasetId = summary.id,
                    config = state.configName,
                    split = state.splitName,
                    columns = state.columns.toList(),
                    maxRows = state.rows,
                    maxBytes = (state.rows.toLong() * 4096L).coerceAtLeast(1_000_000L),
                    destination = destination
                ) { progress ->
                    updateWizard { it.copy(datasetProgress = progress) }
                }
                val license = summary.resolvedLicense()
                val entry = DatasetMixEntry(
                    datasetId = summary.id,
                    config = state.configName,
                    split = state.splitName,
                    columns = state.columns.toList(),
                    weight = state.weight,
                    cacheFile = destination.name,
                    rows = stats.rows,
                    bytes = stats.bytes,
                    license = license,
                    nonCommercial = isLikelyNonCommercial(license)
                )
                updateWizard {
                    it.copy(
                        datasets = it.datasets + entry,
                        selectedDataset = null,
                        structure = null,
                        configName = "",
                        splitName = "",
                        columns = emptySet(),
                        datasetBusy = false,
                        datasetProgress = null,
                        error = null
                    )
                }
            } catch (t: Throwable) {
                updateWizard {
                    it.copy(datasetBusy = false, error = t.message ?: "Download failed.")
                }
            }
        }
    }

    fun removeDataset(index: Int) {
        updateWizard { state ->
            val entry = state.datasets.getOrNull(index) ?: return@updateWizard state
            val id = state.modelId
            if (id != null) {
                val file = File(store.dataDir(id), entry.cacheFile)
                if (file.exists()) file.delete()
            }
            state.copy(datasets = state.datasets.filterIndexed { i, _ -> i != index })
        }
    }

    fun startTraining() {
        val state = _wizard.value
        val id = state.modelId
        val config = state.config
        if (id == null || config == null) return
        if (!ArchitectureRegistry.info(state.architectureId).trainable) {
            updateWizard { it.copy(error = "This architecture is a planning path; native training is not built yet.") }
            return
        }
        if (state.inputModalities != setOf(Modality.TEXT) || state.outputModalities != setOf(Modality.TEXT)) {
            updateWizard { it.copy(error = "Native training currently supports text-to-text only.") }
            return
        }
        if (state.datasets.isEmpty()) {
            updateWizard { it.copy(error = "Add at least one dataset before training.") }
            return
        }
        val bits = state.bits
        val settings = TrainingSettings(
            steps = state.steps,
            batchSize = state.batchSize,
            learningRate = state.learningRate,
            warmupSteps = state.warmupSteps,
            seqLength = state.seqLength,
            checkpointEvery = state.checkpointEvery,
            sampleEvery = state.sampleEvery,
            corpusChars = state.corpusChars,
            autoResume = true
        )
        val manifest = ModelManifest(
            id = id,
            name = state.name.trim().ifBlank { "My model" },
            creator = state.creator.trim(),
            createdAt = System.currentTimeMillis(),
            config = config.copy(
                bits = bits,
                qat = bits in 2..8,
                architectureId = state.architectureId
            ),
            settings = settings,
            datasets = state.datasets,
            tokenizerTrained = false,
            languages = state.languages.toList(),
            behavior = com.modelforge.app.data.BehaviorConfig(
                speedMode = state.speedMode,
                reasoningEnabled = state.reasoningEnabled,
                reasoningRoom = state.reasoningRoom,
                toolCalling = state.toolCalling,
                persona = state.persona.trim()
            )
        )
        store.saveManifest(manifest)
        runCatching { TrainingService.start(getApplication(), id) }
            .onFailure { t ->
                updateWizard {
                    it.copy(error = t.message ?: "Could not start the training service.")
                }
                return
            }
        draftStore.clear()
        refreshModels()
        _wizard.value = WizardState()
        _aiPlanner.value = AiPlannerState()
        _screen.value = Screen.Training(id)
    }

    fun openModel(id: String) {
        _screen.value = Screen.ModelDetail(id)
        refreshModels()
    }

    fun openTraining(id: String) {
        runCatching { TrainingService.start(getApplication(), id) }
        _screen.value = Screen.Training(id)
        refreshModels()
    }

    fun isIgnoringBatteryOptimizations(): Boolean {
        val power = getApplication<Application>().getSystemService(PowerManager::class.java)
            ?: return false
        return power.isIgnoringBatteryOptimizations(getApplication<Application>().packageName)
    }

    fun requestBatteryExemption() {
        val app = getApplication<Application>()
        val direct = Intent(
            Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
            Uri.parse("package:${app.packageName}")
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val fallback = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        runCatching { app.startActivity(direct) }
            .onFailure { runCatching { app.startActivity(fallback) } }
    }

    fun pauseTraining() = TrainingService.pause(getApplication())

    fun resumeTraining() = TrainingService.resume(getApplication())

    fun stopTraining() = TrainingService.stop(getApplication())

    fun openPlayground(id: String) {
        val manifest = store.loadManifest(id) ?: return
        val bits = manifest.config.bits.takeIf { it in 2..8 } ?: 4
        _playground.value = PlaygroundState(
            modelId = id,
            bits = bits,
            speedMode = ResponseModes.normalize(manifest.behavior.speedMode),
            reasoningEnabled = manifest.behavior.reasoningEnabled
        )
        _screen.value = Screen.Playground(id)
    }

    fun checkIdentity(id: String) {
        if (_identityCheck.value.busy) return
        val manifest = store.loadManifest(id) ?: return
        if (manifest.creator.isBlank()) {
            _identityCheck.value = IdentityCheckState(
                modelId = id,
                error = "This model has no creator name, so identity cannot be checked."
            )
            return
        }
        val tokenizerFile = store.tokenizerFile(id)
        if (!tokenizerFile.exists()) {
            _identityCheck.value = IdentityCheckState(
                modelId = id,
                error = "Train this model before checking its identity."
            )
            return
        }
        val checkpoint = store.bestCheckpoint(id)
        if (!checkpoint.exists()) {
            _identityCheck.value = IdentityCheckState(
                modelId = id,
                error = "No best checkpoint yet. Finish a training run first."
            )
            return
        }
        _identityCheck.value = IdentityCheckState(modelId = id, busy = true)
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val tokenizer = BpeTokenizer.load(tokenizerFile)
                val model = NativeModel.load(checkpoint.absolutePath)
                    ?: throw IllegalStateException("Could not load the best checkpoint.")
                try {
                    val questions = IdentityCheck.questions(manifest.languages)
                    if (questions.isEmpty()) {
                        throw IllegalStateException("No identity questions were generated.")
                    }
                    val answers = ArrayList<IdentityAnswer>(questions.size)
                    for (question in questions) {
                        val promptText = "User: $question\nAssistant:"
                        val encoded = tokenizer.encode(promptText)
                        if (encoded.isEmpty()) continue
                        val context = (model.config.contextLength - 4).coerceAtLeast(1)
                        val trimmed = encoded.takeLast(context).toIntArray()
                        val generated = model.generate(
                            trimmed,
                            24,
                            0.2f,
                            20,
                            0.9f,
                            System.nanoTime()
                        )
                        val response = tokenizer.decode(generated)
                        answers += IdentityAnswer(
                            question = question,
                            response = response.trim(),
                            passed = IdentityCheck.isIdentityAnswer(
                                manifest.name,
                                manifest.creator,
                                response
                            )
                        )
                        _identityCheck.value = IdentityCheckState(
                            modelId = id,
                            busy = true,
                            answers = answers.toList()
                        )
                    }
                    _identityCheck.value = IdentityCheckState(
                        modelId = id,
                        answers = answers
                    )
                } finally {
                    model.close()
                }
            } catch (t: Throwable) {
                _identityCheck.value = IdentityCheckState(
                    modelId = id,
                    error = t.message ?: "Identity check failed."
                )
            }
        }
    }

    fun reinforceIdentity(id: String) {
        val manifest = store.loadManifest(id) ?: return
        val live = store.loadLiveState(id)
        if (live?.isActive == true) return
        val updated = manifest.copy(
            settings = manifest.settings.copy(
                steps = manifest.settings.steps + 100,
                identityFraction = manifest.settings.identityFraction.coerceAtLeast(0.08f)
            )
        )
        store.saveManifest(updated)
        refreshModels()
        _identityCheck.value = IdentityCheckState()
        openTraining(id)
    }

    fun writeModelCard(id: String): File? {
        val manifest = store.loadManifest(id) ?: return null
        val card = ModelCard.build(manifest, store.loadLiveState(id))
        val file = store.modelCardFile(id)
        return runCatching {
            file.writeText(card)
            file
        }.getOrNull()
    }

    fun packedFile(id: String, bits: Int): File = store.packedFile(id, bits)

    fun packedFileSize(id: String, bits: Int): Long =
        store.packedFile(id, bits).takeIf { it.exists() }?.length() ?: 0L

    fun updatePlaygroundPrompt(value: String) {
        _playground.update { it.copy(prompt = value) }
    }

    fun updatePlaygroundMaxNew(value: Int) {
        _playground.update { it.copy(maxNew = value.coerceIn(8, 512)) }
    }

    fun updatePlaygroundTemperature(value: Float) {
        _playground.update { it.copy(temperature = value.coerceIn(0.05f, 2f)) }
    }

    fun updatePlaygroundTopK(value: Int) {
        _playground.update { it.copy(topK = value.coerceIn(1, 200)) }
    }

    fun updatePlaygroundTopP(value: Float) {
        _playground.update { it.copy(topP = value.coerceIn(0.05f, 1f)) }
    }

    fun updatePlaygroundMode(mode: String) {
        _playground.update { it.copy(mode = mode, error = null) }
    }

    fun updatePlaygroundBits(bits: Int) {
        if (bits !in intArrayOf(2, 3, 4, 8)) return
        _playground.update { it.copy(bits = bits, exportedBytes = 0L, error = null) }
    }

    fun updatePlaygroundSpeed(mode: String) {
        _playground.update { it.copy(speedMode = ResponseModes.normalize(mode), error = null) }
    }

    fun updatePlaygroundReasoning(enabled: Boolean) {
        _playground.update { it.copy(reasoningEnabled = enabled, thinkingExpanded = enabled) }
    }

    fun togglePlaygroundThinking() {
        _playground.update { it.copy(thinkingExpanded = !it.thinkingExpanded) }
    }

    fun generatePlayground() {
        val state = _playground.value
        if (state.busy || state.modelId.isBlank()) return
        val id = state.modelId
        val tokenizerFile = store.tokenizerFile(id)
        if (!tokenizerFile.exists()) {
            _playground.update {
                it.copy(error = "This model has no tokenizer yet. Train it before generating text.")
            }
            return
        }
        val checkpoint = store.bestCheckpoint(id)
        val packed = store.packedFile(id, state.bits)
        if (state.mode == PlaygroundState.MODE_PACKED && !packed.exists()) {
            _playground.update {
                it.copy(error = "Export a ${state.bits}-bit packed file first.")
            }
            return
        }
        if (state.mode == PlaygroundState.MODE_CHECKPOINT && !checkpoint.exists()) {
            _playground.update {
                it.copy(error = "No best checkpoint yet. Start or finish training first.")
            }
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            _playground.update { it.copy(busy = true, status = "Generating…", error = null) }
            try {
                val tokenizer = BpeTokenizer.load(tokenizerFile)
                val promptText = ResponseModes.prompt(
                    state.prompt,
                    state.reasoningEnabled,
                    state.speedMode
                )
                val encoded = tokenizer.encode(promptText)
                val prompt = if (encoded.isEmpty()) tokenizer.encode(" ") else encoded
                val maxNew = ResponseModes.maxNewTokens(state.maxNew, state.speedMode)
                var text = ""
                var tokens = 0
                var perSecond = 0.0
                var latency = 0L
                if (state.mode == PlaygroundState.MODE_PACKED) {
                    val model = PackedModel.load(packed.absolutePath)
                        ?: throw IllegalStateException("Could not load the packed file. Export it again.")
                    try {
                        val trimmed = prompt.takeLast((model.config.contextLength - 1).coerceAtLeast(1))
                            .toIntArray()
                        val started = System.nanoTime()
                        val generated = model.generate(
                            trimmed, maxNew, state.temperature, state.topK, state.topP, System.nanoTime()
                        )
                        val elapsed = System.nanoTime() - started
                        val seconds = elapsed / 1_000_000_000.0
                        text = tokenizer.decode(generated)
                        tokens = generated.size
                        perSecond = if (seconds > 0.0) generated.size / seconds else 0.0
                        latency = elapsed / 1_000_000L
                    } finally {
                        model.close()
                    }
                } else {
                    val model = NativeModel.load(checkpoint.absolutePath)
                        ?: throw IllegalStateException("Could not load the best checkpoint.")
                    try {
                        val trimmed = prompt.takeLast((model.config.contextLength - 1).coerceAtLeast(1))
                            .toIntArray()
                        val started = System.nanoTime()
                        val generated = model.generate(
                            trimmed, maxNew, state.temperature, state.topK, state.topP, System.nanoTime()
                        )
                        val elapsed = System.nanoTime() - started
                        val seconds = elapsed / 1_000_000_000.0
                        text = tokenizer.decode(generated)
                        tokens = generated.size
                        perSecond = if (seconds > 0.0) generated.size / seconds else 0.0
                        latency = elapsed / 1_000_000L
                    } finally {
                        model.close()
                    }
                }
                _playground.update {
                    it.copy(
                        busy = false,
                        status = "",
                        output = text,
                        generatedTokens = tokens,
                        tokensPerSec = perSecond,
                        latencyMs = latency
                    )
                }
            } catch (t: Throwable) {
                _playground.update {
                    it.copy(
                        busy = false,
                        status = "",
                        error = t.message ?: "Generation failed."
                    )
                }
            }
        }
    }

    fun exportPacked() {
        val state = _playground.value
        if (state.busy || state.modelId.isBlank()) return
        val id = state.modelId
        val bits = state.bits
        val checkpoint = store.bestCheckpoint(id)
        if (!checkpoint.exists()) {
            _playground.update {
                it.copy(error = "No best checkpoint yet. Start or finish training first.")
            }
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            _playground.update {
                it.copy(busy = true, status = "Exporting ${bits}-bit packed model…", error = null)
            }
            try {
                val model = NativeModel.load(checkpoint.absolutePath)
                    ?: throw IllegalStateException("Could not load the best checkpoint.")
                try {
                    val target = store.packedFile(id, bits)
                    if (!model.savePacked(target.absolutePath, bits, 32)) {
                        throw IllegalStateException("The native exporter rejected the ${bits}-bit export.")
                    }
                    _playground.update {
                        it.copy(
                            busy = false,
                            status = "Exported ${target.name}",
                            mode = PlaygroundState.MODE_PACKED,
                            exportedBytes = target.length()
                        )
                    }
                } finally {
                    model.close()
                }
            } catch (t: Throwable) {
                _playground.update {
                    it.copy(
                        busy = false,
                        status = "",
                        error = t.message ?: "Export failed."
                    )
                }
            }
        }
    }

    fun deleteModel(id: String) {
        if (store.loadLiveState(id)?.isActive == true) {
            _notice.value = "Stop training before deleting this model."
            return
        }
        store.deleteModel(id)
        _notice.value = "Model deleted."
        refreshModels()
        _screen.value = Screen.Home
    }

    fun goHome() {
        _screen.value = Screen.Home
        refreshModels()
    }

    fun refreshModels() {
        _models.value = store.listManifests()
    }

    fun saveToken(value: String) {
        tokenStore.token = value
        _tokenPresent.value = tokenStore.token != null
        _notice.value = if (value.isBlank()) "Hugging Face token cleared." else "Token saved securely."
    }

    fun clearToken() {
        tokenStore.clear()
        _tokenPresent.value = false
        _notice.value = "Hugging Face token cleared."
    }

    fun clearNotice() {
        _notice.value = null
    }

    private fun defaultTarget(): Long {
        val cap = maxTargetParams
        return listOf(1_000_000L, 2_000_000L, 5_000_000L, 10_000_000L, 25_000_000L, 50_000_000L)
            .lastOrNull { it <= cap } ?: cap.coerceAtLeast(1_000_000L)
    }

    private fun WizardState.withSettings(config: ModelConfig): WizardState {
        val auto = TrainingSettings.autoFor(config)
        return copy(
            config = config,
            steps = auto.steps,
            batchSize = auto.batchSize,
            seqLength = auto.seqLength,
            learningRate = auto.learningRate,
            warmupSteps = auto.warmupSteps,
            checkpointEvery = auto.checkpointEvery,
            sampleEvery = auto.sampleEvery,
            corpusChars = auto.corpusChars
        )
    }
}
