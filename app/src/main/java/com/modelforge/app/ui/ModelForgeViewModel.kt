package com.modelforge.app.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.os.StatFs
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
import com.modelforge.app.diagnostics.CheckState
import com.modelforge.app.diagnostics.DiagnosticCheck
import com.modelforge.app.diagnostics.DiagnosticsState
import com.modelforge.app.diagnostics.ForgeLog
import com.modelforge.app.nativebridge.NativeModel
import com.modelforge.app.nativebridge.PackedModel
import com.modelforge.app.training.CheckpointHeader
import com.modelforge.app.training.CorpusBudget
import com.modelforge.app.training.TrainingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

sealed class Screen {
    object Home : Screen()
    object Wizard : Screen()
    object Gallery : Screen()
    object Diagnostics : Screen()
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
    val rows: Long = 1000L,
    val availableRows: Long = 0L,
    val rowCountBusy: Boolean = false,
    val weight: Float = 1f,
    val datasetBusy: Boolean = false,
    val datasetProgress: DownloadStats? = null,
    val datasetNotice: String? = null,
    val datasets: List<DatasetMixEntry> = emptyList(),
    val parentId: String = "",
    val seriesId: String = "",
    val seriesVersion: Int = 1,
    val preparing: Boolean = false,
    val error: String? = null
) {
    val canStart: Boolean
        get() = modelId != null && config != null && name.isNotBlank() &&
            creator.isNotBlank() && datasets.isNotEmpty() && !preparing &&
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
    val failedDatasets: List<AiDatasetSuggestion> = emptyList(),
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
    val trainingContention: Boolean = false,
    val cancelled: Boolean = false,
    val exportedBytes: Long = 0L,
    val lastExportFile: String = "",
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
        updateWizard { if (it.datasetBusy) it.copy(datasetNotice = message) else it }
    }
    private val nvidia = NvidiaClient { message ->
        _aiPlanner.update { if (it.busy) it.copy(stage = message) else it }
    }
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

    private val _diagnostics = MutableStateFlow(DiagnosticsState())
    val diagnostics: StateFlow<DiagnosticsState> = _diagnostics.asStateFlow()

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

    fun openDiagnostics() { _screen.value = Screen.Diagnostics }

    fun runDiagnostics() {
        if (_diagnostics.value.running) return
        _diagnostics.value = DiagnosticsState(
            running = true,
            huggingFace = DiagnosticCheck("Hugging Face", CheckState.RUNNING, "Checking public access…"),
            nvidia = DiagnosticCheck("NVIDIA", CheckState.RUNNING, "Checking the saved key…"),
            secureStorage = DiagnosticCheck("Secure storage", CheckState.RUNNING, "Checking encrypted storage…")
        )
        ForgeLog.info("diagnostics", "Manual checks started")
        viewModelScope.launch {
            val secure = DiagnosticCheck(
                "Secure storage",
                if (tokenStore.isAvailable) CheckState.PASS else CheckState.WARN,
                if (tokenStore.isAvailable) {
                    "EncryptedSharedPreferences is available."
                } else {
                    "Unavailable; credentials cannot be stored on this device."
                }
            )
            val hfCheck = checkHuggingFace()
            val nvidiaCheck = checkNvidia()
            _diagnostics.value = DiagnosticsState(
                running = false,
                huggingFace = hfCheck,
                nvidia = nvidiaCheck,
                secureStorage = secure,
                lastRun = System.currentTimeMillis()
            )
            ForgeLog.info(
                "diagnostics",
                "Checks finished: Hugging Face=${hfCheck.state.label}, NVIDIA=${nvidiaCheck.state.label}, " +
                    "secure storage=${secure.state.label}"
            )
        }
    }

    private suspend fun checkHuggingFace(): DiagnosticCheck {
        val token = tokenStore.token
        return try {
            if (token == null) {
                val results = hf.search("TinyStories", 1)
                DiagnosticCheck(
                    "Hugging Face",
                    CheckState.PASS,
                    "No token saved; public dataset search is reachable (${results.size} result)."
                )
            } else {
                DiagnosticCheck("Hugging Face", CheckState.PASS, hf.validateToken())
            }
        } catch (t: kotlinx.coroutines.CancellationException) {
            throw t
        } catch (t: Throwable) {
            DiagnosticCheck(
                "Hugging Face",
                CheckState.FAIL,
                t.message ?: "Hugging Face check failed."
            )
        }
    }

    private suspend fun checkNvidia(): DiagnosticCheck {
        val key = tokenStore.nvidiaToken
        if (key == null) {
            return DiagnosticCheck(
                "NVIDIA",
                CheckState.INFO,
                "No key saved; AI planning is optional."
            )
        }
        return try {
            DiagnosticCheck("NVIDIA", CheckState.PASS, nvidia.validateKey(key))
        } catch (t: kotlinx.coroutines.CancellationException) {
            throw t
        } catch (t: Throwable) {
            DiagnosticCheck(
                "NVIDIA",
                CheckState.FAIL,
                t.message ?: "NVIDIA check failed."
            )
        }
    }

    fun manifest(id: String): ModelManifest? = store.loadManifest(id)

    fun liveState(id: String): TrainingLiveState? = store.loadLiveState(id)

    @Volatile
    private var playgroundCancel = false

    fun checkpointMegabytes(id: String): Pair<Long, Long> {
        // Round up: a valid checkpoint smaller than 1 MB must not read as 0
        // (the playground treated 0 as "missing").
        fun mb(file: File): Long =
            if (file.exists()) (file.length() + 1024L * 1024L - 1L) / (1024L * 1024L) else 0L
        return mb(store.lastCheckpoint(id)) to mb(inferenceCheckpoint(id))
    }

    /**
     * The checkpoint to load for testing/export. Prefer the best checkpoint,
     * but fall back to the final one: v2.3.0 could stop before its first
     * periodic checkpoint and leave only last.mfck behind.
     */
    private fun inferenceCheckpoint(id: String): File {
        val best = store.bestCheckpoint(id)
        if (CheckpointHeader.hasMagic(best)) return best
        val last = store.lastCheckpoint(id)
        if (CheckpointHeader.hasMagic(last)) return last
        return best
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
        val context = draft?.contextLength?.takeIf { it > 0 }?.coerceIn(128, 1_000_000)
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
                contextChoice = context ?: 0,
                languages = draft.languages.toSet(),
                speedMode = draft.speedMode,
                reasoningEnabled = draft.reasoningEnabled,
                reasoningRoom = draft.reasoningRoom,
                toolCalling = draft.toolCalling,
                persona = draft.persona,
                bits = draft.bits,
                steps = draft.steps.coerceAtLeast(1),
                batchSize = draft.batchSize.coerceIn(1, 64),
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
        val bounded = length.coerceIn(128, 1_000_000)
        val config = ArchitectureRegistry.derive(
            state.architectureId,
            state.targetParams,
            maxTargetParams,
            bounded
        ).config
        updateWizard {
            it.copy(contextChoice = bounded, config = config ?: it.config, error = null)
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
        val saved = tokenStore.saveNvidiaToken(value)
        _aiKeyPresent.value = tokenStore.nvidiaToken != null
        ForgeLog.info("credentials", "NVIDIA key saved=${tokenStore.nvidiaToken != null} ok=$saved")
        _aiPlanner.value = AiPlannerState(
            error = when {
                !saved -> "The key could not be saved securely. Try again or restart the app."
                value.isBlank() -> "NVIDIA API key cleared."
                !_aiKeyPresent.value -> "The key could not be saved securely."
                else -> null
            }
        )
    }

    fun clearAiKey() {
        tokenStore.clearNvidia()
        _aiKeyPresent.value = false
        ForgeLog.info("credentials", "NVIDIA key cleared")
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
        val failed = _aiPlanner.value.failedDatasets.ifEmpty {
            val reasons = _aiPlanner.value.skippedDatasetReasons
            plan.datasets.filter { suggestion ->
                suggestion.datasetId in _aiPlanner.value.skippedDatasetIds &&
                    !reasons[suggestion.datasetId].orEmpty().contains("Already in the mix")
            }
        }
        if (failed.isEmpty()) return
        _aiPlanner.value = _aiPlanner.value.copy(
            busy = true,
            streaming = "",
            applied = false,
            stage = "Retrying ${failed.size} failed dataset(s)",
            skippedDatasetIds = emptySet(),
            skippedDatasetReasons = emptyMap(),
            startedAt = System.currentTimeMillis(),
            error = null
        )
        aiJob = viewModelScope.launch {
            try {
                autoApplyDatasets(plan, req, only = failed)
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
            val steps = training.steps.coerceAtLeast(1)
            state.copy(
                steps = steps,
                batchSize = training.batchSize.coerceIn(1, 64),
                seqLength = seq.coerceAtLeast(32),
                learningRate = training.learningRate.coerceIn(1e-5f, 1e-2f),
                warmupSteps = training.warmupSteps.coerceIn(1, steps),
                checkpointEvery = maxOf(20, steps / 10),
                sampleEvery = maxOf(20, steps / 5),
                corpusChars = CorpusBudget.resolve(
                    requested = 0,
                    steps = steps,
                    batchSize = training.batchSize,
                    seqLength = seq,
                    availableRamBytes = HardwareDetector.availableRamBytes(getApplication<Application>())
                ),
                error = null
            )
        }
    }

    private suspend fun autoApplyDatasets(
        plan: AiPlan,
        request: AiPlanRequest,
        only: List<AiDatasetSuggestion>? = null
    ) {
        val modelId = _wizard.value.modelId ?: return
        val dataDir = store.dataDir(modelId)
        aiRateLimited = false
        val suggestions = only ?: plan.datasets
        val attempted = HashSet<String>(suggestions.map { it.datasetId })
        for ((position, suggestion) in suggestions.withIndex()) {
            currentCoroutineContext().ensureActive()
            applyWithRetries(suggestion, "${position + 1}/${suggestions.size}", dataDir, modelId)
            if (aiRateLimited) break
        }

        // A model needs complementary data, not one source repeated. If the AI
        // returned fewer than three distinct ids (or several downloads failed),
        // top the mix up with verified fallbacks for the requested capabilities.
        if (only == null) {
            val applied = _wizard.value.datasets.map { it.datasetId }.distinct()
            if (applied.size < 3) {
                val fallbacks = AiDatasetFallbacks.forRequest(
                    languages = request.languages,
                    toolCalling = request.toolCalling,
                    reasoning = request.reasoning
                ).filter { fallback ->
                    fallback.datasetId !in applied && fallback.datasetId !in attempted
                }
                for ((position, fallback) in fallbacks.withIndex()) {
                    if (aiRateLimited) break
                    currentCoroutineContext().ensureActive()
                    _aiPlanner.update {
                        it.copy(stage = "Adding complementary data ${position + 1}/${fallbacks.size}")
                    }
                    if (applyWithRetries(
                            fallback,
                            "complementary ${position + 1}/${fallbacks.size}",
                            dataDir,
                            modelId
                        )
                    ) {
                        attempted += fallback.datasetId
                        if (_wizard.value.datasets.map { it.datasetId }.distinct().size >= 3) break
                    }
                }
            }
        }

        // "Already in the mix" is not a download failure, so it must not show
        // up as something the Retry button can fix.
        val reasons = _aiPlanner.value.skippedDatasetReasons
        val failed = suggestions.filter { suggestion ->
            suggestion.datasetId in _aiPlanner.value.skippedDatasetIds &&
                !reasons[suggestion.datasetId].orEmpty().contains("Already in the mix")
        }
        _aiPlanner.update { it.copy(failedDatasets = failed) }
    }

    /**
     * Three automatic attempts per AI-chosen dataset. Transient Hugging Face
     * failures, a slow page or a rate-limit hiccup should not silently drop a
     * dataset; the caller exposes a Retry button for whatever still fails.
     */
    private suspend fun applyWithRetries(
        suggestion: AiDatasetSuggestion,
        label: String,
        dataDir: File,
        modelId: String
    ): Boolean {
        repeat(3) { attempt ->
            currentCoroutineContext().ensureActive()
            if (tryApplyDataset(suggestion, label, dataDir, modelId)) return true
            if (aiRateLimited) return false
            if (attempt < 2) {
                _aiPlanner.update {
                    it.copy(stage = "Retrying ${suggestion.datasetId} (${attempt + 2}/3)")
                }
                delay(1_200L * (attempt + 1))
            }
        }
        return false
    }

    private suspend fun tryApplyDataset(
        suggestion: AiDatasetSuggestion,
        label: String,
        dataDir: File,
        modelId: String
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
            val rows = suggestion.rows.coerceAtLeast(1)
            val maxBytes = minOf(rows.toLong() * 2048L, freeDownloadBytes(modelId))
            _aiPlanner.update { it.copy(stage = "Downloading $id ($label)") }
            val stats = hf.downloadText(
                datasetId = id,
                config = resolved.config,
                split = resolved.split,
                columns = resolved.columns,
                maxRows = rows,
                maxBytes = maxBytes,
                destination = destination,
                knownTotalRows = resolved.totalRows
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
        val steps = value.coerceAtLeast(1)
        updateWizard {
            it.copy(
                steps = steps,
                warmupSteps = maxOf(10, steps / 20),
                checkpointEvery = maxOf(20, steps / 10),
                sampleEvery = maxOf(20, steps / 5),
                corpusChars = CorpusBudget.resolve(
                    requested = it.corpusChars,
                    steps = steps,
                    batchSize = it.batchSize,
                    seqLength = it.seqLength,
                    availableRamBytes = HardwareDetector.availableRamBytes(getApplication<Application>())
                )
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
        val presetContext = preset.contextLength.takeIf { it > 0 }
            ?.coerceIn(128, 1_000_000)
        val derive = ArchitectureRegistry.derive(
            preset.architectureId,
            preset.targetParams.coerceIn(1_000_000L, maxTargetParams),
            maxTargetParams,
            presetContext
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
                contextChoice = presetContext ?: 0,
                languages = preset.languages.toSet(),
                speedMode = preset.speedMode,
                reasoningEnabled = preset.reasoningEnabled,
                reasoningRoom = preset.reasoningRoom,
                toolCalling = preset.toolCalling,
                persona = preset.persona,
                bits = if (preset.bits in info.supportedBits) preset.bits else info.supportedBits.firstOrNull() ?: 4,
                steps = preset.steps.coerceAtLeast(1),
                batchSize = preset.batchSize.coerceIn(1, 64),
                seqLength = preset.seqLength.coerceAtLeast(32),
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
                availableRows = 0L,
                rowCountBusy = false,
                datasetBusy = true,
                datasetProgress = null,
                error = null
            )
        }
        viewModelScope.launch {
            try {
                val structure = hf.structure(summary.id)
                val first = structure.configs.firstOrNull()
                val firstSplit = first?.splits?.firstOrNull().orEmpty()
                val firstRows = first?.splitRows?.get(firstSplit) ?: 0L
                updateWizard {
                    // The user may have picked another dataset while this
                    // request was in flight; never overwrite the newer choice.
                    if (it.selectedDataset?.id != summary.id) return@updateWizard it
                    it.copy(
                        structure = structure,
                        configName = first?.name.orEmpty(),
                        splitName = firstSplit,
                        columns = first?.textColumns?.toSet() ?: emptySet(),
                        availableRows = firstRows,
                        rowCountBusy = first != null && firstRows <= 0L,
                        datasetBusy = false,
                        error = if (structure.configs.isEmpty()) {
                            "No configs were found for ${summary.id}."
                        } else null
                    )
                }
                if (first != null && firstRows <= 0L) refreshRowCount()
            } catch (t: Throwable) {
                updateWizard {
                    if (it.selectedDataset?.id != summary.id) return@updateWizard it
                    it.copy(
                        availableRows = 0L,
                        rowCountBusy = false,
                        datasetBusy = false,
                        error = t.message ?: "Could not inspect this dataset."
                    )
                }
            }
        }
    }

    private fun refreshRowCount() {
        val state = _wizard.value
        val summary = state.selectedDataset ?: return
        val config = state.configName
        val split = state.splitName
        if (config.isBlank() || split.isBlank()) return
        val known = state.structure?.configs
            ?.firstOrNull { it.name == config }
            ?.splitRows?.get(split)
            ?.takeIf { it > 0L }
        if (known != null) {
            updateWizard {
                if (it.selectedDataset?.id == summary.id &&
                    it.configName == config && it.splitName == split
                ) {
                    it.copy(availableRows = known, rowCountBusy = false)
                } else it
            }
            return
        }
        updateWizard { it.copy(availableRows = 0L, rowCountBusy = true) }
        viewModelScope.launch {
            val total = try {
                hf.rowCount(summary.id, config, split)
            } catch (t: kotlinx.coroutines.CancellationException) {
                throw t
            } catch (t: Throwable) {
                ForgeLog.warn("dataset", "Could not read split size for ${summary.id}", t)
                null
            }
            updateWizard {
                if (it.selectedDataset?.id == summary.id &&
                    it.configName == config && it.splitName == split
                ) {
                    it.copy(
                        availableRows = total ?: 0L,
                        rowCountBusy = false
                    )
                } else it
            }
        }
    }

    fun selectConfig(name: String) {
        updateWizard { state ->
            val cfg = state.structure?.configs?.firstOrNull { it.name == name }
            val firstSplit = cfg?.splits?.firstOrNull().orEmpty()
            state.copy(
                configName = name,
                splitName = firstSplit,
                columns = cfg?.textColumns?.toSet() ?: emptySet(),
                availableRows = cfg?.splitRows?.get(firstSplit) ?: 0L
            )
        }
        refreshRowCount()
    }

    fun selectSplit(name: String) {
        updateWizard { state ->
            val known = state.structure?.configs
                ?.firstOrNull { it.name == state.configName }
                ?.splitRows?.get(name)
                ?: 0L
            state.copy(splitName = name, availableRows = known)
        }
        refreshRowCount()
    }

    fun toggleColumn(name: String) {
        updateWizard { state ->
            val columns = if (name in state.columns) state.columns - name else state.columns + name
            state.copy(columns = columns)
        }
    }

    fun updateRows(value: Long) {
        updateWizard { it.copy(rows = value.coerceAtLeast(1L), error = null) }
    }

    fun useEntireSplit() {
        val total = _wizard.value.availableRows
        if (total <= 0L) {
            updateWizard { it.copy(error = "The split row count is still unknown. Try again in a moment.") }
            return
        }
        updateRows(total)
    }

    /**
     * Weight is a mixing multiplier, not a percentage. The old 0.1x-5x clamp
     * made it impossible to ask for a strongly dominant dataset; only a
     * non-positive value is rejected because it would silently drop the data.
     */
    fun updateWeight(value: Float) {
        if (!value.isFinite()) return
        updateWizard { it.copy(weight = value.coerceIn(0.01f, 1000f)) }
    }

    fun downloadSelectedDataset() {
        val state = _wizard.value
        val modelId = state.modelId ?: return
        val summary = state.selectedDataset ?: return
        if (state.datasetBusy) return
        if (state.configName.isBlank() || state.splitName.isBlank() || state.columns.isEmpty()) {
            updateWizard {
                it.copy(error = "Choose a config, split and at least one text column.")
            }
            return
        }
        if (state.datasets.any {
                it.datasetId == summary.id && it.config == state.configName &&
                    it.split == state.splitName
            }
        ) {
            updateWizard {
                it.copy(error = "This dataset, config and split are already in the mix.")
            }
            return
        }
        // A UUID keeps two downloads of the same dataset, and repeated
        // add/remove cycles, from ever sharing or overwriting a cache file.
        val destination = File(
            store.dataDir(modelId),
            "dataset_${UUID.randomUUID()}.txt"
        )
        val requestedRows = state.rows
        val requestedTotalRows = state.availableRows
        val requestedConfig = state.configName
        val requestedSplit = state.splitName
        val requestedColumns = state.columns.toList()
        val requestedWeight = state.weight
        viewModelScope.launch {
            updateWizard { it.copy(datasetBusy = true, datasetProgress = null, datasetNotice = null, error = null) }
            try {
                val stats = hf.downloadText(
                    datasetId = summary.id,
                    config = requestedConfig,
                    split = requestedSplit,
                    columns = requestedColumns,
                    maxRows = requestedRows,
                    maxBytes = freeDownloadBytes(modelId),
                    destination = destination,
                    knownTotalRows = requestedTotalRows
                ) { progress ->
                    updateWizard { it.copy(datasetProgress = progress) }
                }
                val license = summary.resolvedLicense()
                val entry = DatasetMixEntry(
                    datasetId = summary.id,
                    config = requestedConfig,
                    split = requestedSplit,
                    columns = requestedColumns,
                    weight = requestedWeight,
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
                        availableRows = 0L,
                        rowCountBusy = false,
                        datasetBusy = false,
                        datasetProgress = null,
                        datasetNotice = null,
                        error = null
                    )
                }
            } catch (t: Throwable) {
                // A partial file must never be left behind: a later resume or
                // changed row count would otherwise mix truncated text.
                runCatching { destination.delete() }
                updateWizard {
                    it.copy(
                        datasetBusy = false,
                        datasetProgress = null,
                        datasetNotice = null,
                        error = t.message ?: "Download failed."
                    )
                }
            }
        }
    }

    fun removeDataset(index: Int) {
        updateWizard { state ->
            val entry = state.datasets.getOrNull(index) ?: return@updateWizard state
            val remaining = state.datasets.filterIndexed { i, _ -> i != index }
            val id = state.modelId
            if (id != null && remaining.none { it.cacheFile == entry.cacheFile }) {
                runCatching { File(store.dataDir(id), entry.cacheFile).delete() }
            }
            state.copy(datasets = remaining)
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
            tokenizerTrained = state.parentId.isNotBlank(),
            languages = state.languages.toList(),
            behavior = com.modelforge.app.data.BehaviorConfig(
                speedMode = state.speedMode,
                reasoningEnabled = state.reasoningEnabled,
                reasoningRoom = state.reasoningRoom,
                toolCalling = state.toolCalling,
                persona = state.persona.trim()
            ),
            seriesId = state.seriesId.ifBlank { id },
            seriesVersion = state.seriesVersion,
            parentId = state.parentId
        )
        store.saveManifest(manifest)
        val parentId = state.parentId
        if (parentId.isNotBlank()) {
            updateWizard {
                it.copy(
                    preparing = true,
                    error = null,
                    datasetNotice = "Copying weights and tokenizer from the previous version…"
                )
            }
            viewModelScope.launch(Dispatchers.IO) {
                val seeded = runCatching { seedFromParent(parentId, id) }.getOrDefault(false)
                if (!seeded) {
                    updateWizard {
                        it.copy(
                            preparing = false,
                            datasetNotice = null,
                            error = "No readable checkpoint was found in the previous version. " +
                                "Train that version first, or start a model from scratch."
                        )
                    }
                    return@launch
                }
                launchTraining(id)
            }
            return
        }
        launchTraining(id)
    }

    private fun launchTraining(id: String) {
        runCatching { TrainingService.start(getApplication(), id) }
            .onFailure { t ->
                updateWizard {
                    it.copy(
                        preparing = false,
                        datasetNotice = null,
                        error = t.message ?: "Could not start the training service."
                    )
                }
                return
            }
        draftStore.clear()
        refreshModels()
        _wizard.value = WizardState()
        _aiPlanner.value = AiPlannerState()
        _screen.value = Screen.Training(id)
    }

    /**
     * Copies the parent's best weights and tokenizer into the new version.
     * Training reads them once, then writes its own checkpoints, so the parent
     * stays reproducible and can still be evaluated side by side.
     */
    private fun seedFromParent(parentId: String, id: String): Boolean {
        val source = listOf(
            store.bestCheckpoint(parentId),
            store.lastCheckpoint(parentId),
            store.recoveryCheckpoint(parentId)
        ).firstOrNull { CheckpointHeader.hasMagic(it) } ?: return false
        source.copyTo(store.initCheckpoint(id), overwrite = true)
        val tokenizer = store.tokenizerFile(parentId)
        if (tokenizer.exists()) {
            tokenizer.copyTo(store.tokenizerFile(id), overwrite = true)
        }
        return true
    }

    /**
     * Starts the next version in a series. The wizard is pre-filled with the
     * parent's architecture, datasets and behaviour, so the user only changes
     * what this version should improve: extra datasets, more steps, a new name.
     */
    fun startNextVersion(parentId: String) {
        val parent = store.loadManifest(parentId) ?: return
        val version = parent.seriesVersion + 1
        val baseName = parent.name.replace(Regex("\\s+v\\d+$"), "")
        val name = "$baseName v$version"
        val id = store.newModelId(name)
        val weakest = runCatching { bench.latestBySource()[parentId]?.weakestPart }.getOrNull()
        val needsMoreTraining = weakest == null || weakest in setOf(
            "Quality", "Calibration", "Robustness", "Generation"
        )
        val nextSteps = if (needsMoreTraining) {
            (parent.settings.steps * 2).coerceAtMost(20000)
        } else {
            parent.settings.steps
        }
        val nextCorpus = if (needsMoreTraining) {
            (parent.settings.corpusChars * 2).coerceAtMost(24_000_000)
        } else {
            parent.settings.corpusChars
        }
        _wizard.value = WizardState(
            modelId = id,
            name = name,
            creator = parent.creator,
            architectureId = parent.config.architectureId,
            targetParams = parent.config.paramCount(),
            inputModalities = setOf(Modality.TEXT),
            outputModalities = setOf(Modality.TEXT),
            languages = parent.languages.toSet().ifEmpty { setOf("English") },
            speedMode = parent.behavior.speedMode,
            reasoningEnabled = parent.behavior.reasoningEnabled,
            reasoningRoom = parent.behavior.reasoningRoom,
            toolCalling = parent.behavior.toolCalling,
            persona = parent.behavior.persona,
            config = parent.config,
            bits = if (parent.config.bits in 2..8) parent.config.bits else 4,
            steps = nextSteps,
            batchSize = parent.settings.batchSize,
            seqLength = parent.settings.seqLength,
            learningRate = parent.settings.learningRate / 2f,
            warmupSteps = parent.settings.warmupSteps,
            checkpointEvery = parent.settings.checkpointEvery,
            sampleEvery = parent.settings.sampleEvery,
            corpusChars = nextCorpus,
            datasets = parent.datasets,
            parentId = parent.id,
            seriesId = parent.lineageId,
            seriesVersion = version
        )
        _screen.value = Screen.Wizard
    }

    data class SeriesEntry(
        val manifest: ModelManifest,
        val score: com.modelforge.app.data.bench.BenchClient.RunSummary?
    )

    private val bench by lazy { com.modelforge.app.data.bench.BenchClient(getApplication()) }

    fun benchAvailability(): com.modelforge.app.data.bench.BenchClient.Availability =
        bench.availability()

    fun refreshBenchAvailability() = benchAvailability()

    /** Every version of the series with its newest completed Bench score, if any. */
    fun seriesWithScores(modelId: String): List<SeriesEntry> {
        val scores = runCatching { bench.latestBySource() }.getOrDefault(emptyMap())
        return seriesFor(modelId).map { entry ->
            SeriesEntry(entry, scores[entry.id])
        }
    }

    /**
     * Turns measured weaknesses into concrete next-version advice. This is
     * deliberately heuristic and transparent: the app shows which Bench number
     * produced each suggestion instead of claiming an AI decided in secret.
     */
    fun seriesAdvice(modelId: String): List<String> {
        val entries = seriesWithScores(modelId)
        val current = entries.firstOrNull { it.manifest.id == modelId } ?: return emptyList()
        val advice = ArrayList<String>()
        val score = current.score
        if (score == null) {
            advice += "No completed Bench run for this version yet. Run one to get advice based on measurements."
            return advice
        }
        advice += "Latest Bench: ${String.format("%.0f", score.composite)}/100 (${score.grade}) · ${score.verdict}."
        val previous = entries
            .filter { it.manifest.seriesVersion < current.manifest.seriesVersion }
            .maxByOrNull { it.manifest.seriesVersion }
        previous?.score?.let { before ->
            val delta = score.composite - before.composite
            advice += if (delta >= 0) {
                "Up ${String.format("%.1f", delta)} points from v${previous.manifest.seriesVersion}."
            } else {
                "Down ${String.format("%.1f", -delta)} points from v${previous.manifest.seriesVersion}; " +
                    "keep that version's checkpoint before training further."
            }
        }
        when (score.weakestPart) {
            "Quality" -> advice +=
                "Quality is the weakest part (${String.format("%.0f", score.weakestScore)}/100). " +
                "The next version starts with twice the steps and corpus, at half the learning rate."
            "Calibration" -> advice +=
                "Confidence is mis-calibrated. Train longer on a broader mix rather than " +
                "repeating one dataset, so confident wrong answers become rarer."
            "Robustness" -> advice +=
                "Answers change when choices are reordered. Add more question/answer formats " +
                "and avoid a single narrow dataset for the next run."
            "Generation" -> advice +=
                "Repetition is the weak point. Prefer longer, varied prose and remove duplicated rows."
            "Efficiency" -> advice +=
                "Decode speed is the weak point. Architecture stays fixed when seeding weights, " +
                "so export 4-bit packed for speed, or start a smaller model from scratch."
            "Memory" -> advice +=
                "Peak memory is the weak point. Use a 4-bit export, or design a smaller architecture " +
                "in a fresh model; seeded versions must keep the same shape."
            "Cost" -> advice +=
                "Cost per million tokens is high. Fewer parameters or a packed export improves it; " +
                "the Bench cost settings should match your electricity price."
            else -> advice += "No weak section stood out. A longer run on fresh data is the safest next step."
        }
        return advice
    }

    /** All versions of one lineage, oldest first, for the series view. */
    fun seriesFor(modelId: String): List<ModelManifest> {
        val current = store.loadManifest(modelId) ?: return emptyList()
        return store.listManifests()
            .filter { it.lineageId == current.lineageId }
            .sortedBy { it.seriesVersion }
    }

    fun openModel(id: String) {
        _screen.value = Screen.ModelDetail(id)
        refreshModels()
    }

    fun openTraining(id: String) {
        // On a finished model this adds another run of the configured length
        // instead of reporting that there is nothing left to do.
        runCatching { TrainingService.start(getApplication(), id, extend = true) }
        _screen.value = Screen.Training(id)
        refreshModels()
    }

    /**
     * Continues the same model for exactly [extraSteps] more optimizer steps,
     * resuming weights and AdamW state from its latest checkpoint.
     */
    fun extendTraining(modelId: String, extraSteps: Int) {
        val manifest = store.loadManifest(modelId) ?: return
        val steps = extraSteps.coerceIn(10, 20000)
        store.saveManifest(
            manifest.copy(settings = manifest.settings.copy(steps = steps))
        )
        runCatching { TrainingService.start(getApplication(), modelId, extend = true) }
        _screen.value = Screen.Training(modelId)
        refreshModels()
    }

    /**
     * Builds the free Colab package: token cache, config, tokenizer, optional
     * starting weights and a generated notebook. Drive is used as the transfer
     * folder; there is no hidden Colab automation and no stored credential.
     */
    fun exportCloudPackage(modelId: String, onResult: (File?, String?) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = runCatching {
                val manifest = store.loadManifest(modelId)
                    ?: throw IllegalStateException("This model no longer exists.")
                val tokens = store.tokenCacheFile(modelId)
                if (!tokens.isFile) {
                    throw IllegalStateException(
                        "No token cache yet. Start local training once so the corpus is " +
                            "prepared, then export the cloud package."
                    )
                }
                val dir = File(getApplication<Application>().cacheDir, "cloud").apply { mkdirs() }
                val safeName = manifest.name.replace(Regex("[^A-Za-z0-9._-]+"), "-").take(40)
                val zip = File(dir, "ModelForge-$safeName-colab.zip")
                java.util.zip.ZipOutputStream(zip.outputStream().buffered()).use { out ->
                    fun put(name: String, write: (java.io.OutputStream) -> Unit) {
                        out.putNextEntry(java.util.zip.ZipEntry(name))
                        write(out)
                        out.closeEntry()
                    }
                    fun putFile(name: String, file: File) = put(name) { target ->
                        file.inputStream().use { it.copyTo(target) }
                    }
                    fun putText(name: String, text: String) = put(name) { target ->
                        target.write(text.toByteArray(Charsets.UTF_8))
                    }
                    putFile("input/tokens.bin", tokens)
                    val configFile = store.configFile(modelId)
                    if (configFile.isFile) {
                        putFile("input/config.json", configFile)
                    } else {
                        val c = manifest.config
                        putText(
                            "input/config.json",
                            """{"vocabSize": ${c.vocabSize}, "contextLength": ${c.contextLength}, """ +
                                """"dModel": ${c.dModel}, "layers": ${c.layers}, """ +
                                """"heads": ${c.heads}, "dFf": ${c.dFf}}"""
                        )
                    }
                    val tokenizer = store.tokenizerFile(modelId)
                    if (tokenizer.isFile) putFile("input/tokenizer.json", tokenizer)
                    val best = store.bestCheckpoint(modelId)
                    if (CheckpointHeader.hasMagic(best)) putFile("input/init.mfck", best)
                    putText("ModelForge-$safeName.ipynb", com.modelforge.app.cloud.ColabNotebook.notebook(manifest))
                    putText(
                        "README.txt",
                        "ModelForge cloud training package for ${manifest.name}\n" +
                            "\n" +
                            "1. Upload the contents of this zip to Google Drive: My Drive/ModelForge/${manifest.id}/\n" +
                            "   so that input/tokens.bin and the .ipynb sit exactly there.\n" +
                            "2. Open the .ipynb in Google Colab (CPU runtime is enough; the engine is CPU-only).\n" +
                            "3. Run the cells. Progress and checkpoints are mirrored to output/ in the same folder.\n" +
                            "4. In ModelForge, use Import cloud checkpoint and choose output/best.mfck.\n" +
                            "\n" +
                            "Colab sessions are managed by Google in your browser. The app does not log in to\n" +
                            "your Google account and cannot start or stop a session for you.\n"
                    )
                }
                zip
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { onResult(it, null) }
                result.onFailure { onResult(null, it.message ?: "The cloud package could not be written.") }
            }
        }
    }

    /** Imports a Drive-downloaded .mfck as this model's best checkpoint. */
    fun importCloudCheckpoint(modelId: String, uri: android.net.Uri, onResult: (Boolean, String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val manifest = store.loadManifest(modelId)
                    ?: throw IllegalStateException("This model no longer exists.")
                val tmp = File(store.checkpointsDir(modelId), ".cloud-import.mfck")
                getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output) }
                } ?: throw IllegalStateException("The selected file could not be opened.")
                val config = CheckpointHeader.readConfig(tmp)
                    ?: throw IllegalStateException("That is not a complete ModelForge .mfck checkpoint.")
                val expected = manifest.config
                val matches = config.vocabSize == expected.vocabSize &&
                    config.contextLength == expected.contextLength &&
                    config.dModel == expected.dModel && config.layers == expected.layers &&
                    config.heads == expected.heads && config.dFf == expected.dFf &&
                    config.tiedEmbeddings == expected.tiedEmbeddings
                if (!matches) {
                    tmp.delete()
                    throw IllegalStateException(
                        "The checkpoint architecture does not match ${manifest.name}. " +
                            "Import it into the model it was trained for."
                    )
                }
                tmp.copyTo(store.bestCheckpoint(modelId), overwrite = true)
                tmp.delete()
                withContext(Dispatchers.Main) {
                    onResult(true, "Cloud checkpoint imported. Open the playground to test and export it.")
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) {
                    onResult(false, t.message ?: "The checkpoint could not be imported.")
                }
            }
        }
    }

    /** Reads a Drive-downloaded progress.json written by the notebook. */
    fun readCloudProgress(uri: android.net.Uri, onResult: (String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val text = runCatching {
                val raw = getApplication<Application>().contentResolver.openInputStream(uri)
                    ?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }
                    ?: throw IllegalStateException("The selected file could not be opened.")
                val json = org.json.JSONObject(raw)
                val step = json.optInt("step")
                val total = json.optInt("totalSteps")
                val loss = json.optDouble("loss")
                val best = json.optDouble("bestLoss")
                val tps = json.optDouble("tokensPerSec")
                val eta = json.optLong("etaSeconds")
                val status = json.optString("status", "unknown")
                "Cloud run: $status · step $step/$total · loss " +
                    String.format("%.4f", loss) + " · best " + String.format("%.4f", best) +
                    " · " + String.format("%.0f tok/s", tps) +
                    if (eta > 0) " · ETA ${eta / 60} min" else ""
            }.getOrElse { it.message ?: "progress.json could not be read." }
            withContext(Dispatchers.Main) { onResult(text) }
        }
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
        val checkpoint = inferenceCheckpoint(id)
        if (!CheckpointHeader.hasMagic(checkpoint)) {
            _identityCheck.value = IdentityCheckState(
                modelId = id,
                error = "No checkpoint yet. Finish or stop a training run first."
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
        val checkpoint = inferenceCheckpoint(id)
        val packed = store.packedFile(id, state.bits)
        if (state.mode == PlaygroundState.MODE_PACKED && !packed.exists()) {
            _playground.update {
                it.copy(error = "Export a ${state.bits}-bit packed file first.")
            }
            return
        }
        if (state.mode == PlaygroundState.MODE_CHECKPOINT && !CheckpointHeader.hasMagic(checkpoint)) {
            _playground.update {
                it.copy(error = "No checkpoint yet. Start or finish training first.")
            }
            return
        }
        // Generation and training share the same cores. Say so instead of
        // letting a slow answer look like a hang.
        val trainingActive = store.loadLiveState(id)?.isActive == true
        viewModelScope.launch(Dispatchers.Default) {
            playgroundCancel = false
            _playground.update {
                it.copy(
                    busy = true,
                    status = "Loading model…",
                    output = "",
                    generatedTokens = 0,
                    tokensPerSec = 0.0,
                    latencyMs = 0L,
                    trainingContention = trainingActive,
                    cancelled = false,
                    error = null
                )
            }
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
                val streamed = ArrayList<Int>(maxNew)
                var started = 0L
                var lastUi = 0L

                // Runs on the native thread. UI updates are throttled so text
                // rendering never becomes the bottleneck; cancellation is read
                // between tokens, which is what makes Stop immediate.
                val onToken: (Int) -> Boolean = { token ->
                    streamed.add(token)
                    val now = System.nanoTime()
                    if (streamed.size == 1 || now - lastUi > 120_000_000L) {
                        lastUi = now
                        val seconds = (now - started) / 1_000_000_000.0
                        val perSecond = if (seconds > 0.0) streamed.size / seconds else 0.0
                        val decoded = tokenizer.decode(streamed.toIntArray())
                        _playground.update {
                            it.copy(
                                status = "${streamed.size}/$maxNew tokens · " +
                                    String.format("%.1f tok/s", perSecond),
                                output = decoded,
                                generatedTokens = streamed.size,
                                tokensPerSec = perSecond,
                                latencyMs = (now - started) / 1_000_000L
                            )
                        }
                    }
                    !playgroundCancel
                }

                val generated: IntArray
                val elapsed: Long
                if (state.mode == PlaygroundState.MODE_PACKED) {
                    val model = PackedModel.load(packed.absolutePath)
                        ?: throw IllegalStateException("Could not load the packed file. Export it again.")
                    try {
                        val trimmed = prompt.takeLast((model.config.contextLength - 1).coerceAtLeast(1))
                            .toIntArray()
                        _playground.update {
                            it.copy(status = "Prefilling ${trimmed.size} prompt tokens…")
                        }
                        val t0 = System.nanoTime()
                        started = t0
                        generated = model.generateStreaming(
                            trimmed, maxNew, state.temperature, state.topK, state.topP,
                            System.nanoTime(), onToken
                        )
                        elapsed = System.nanoTime() - t0
                    } finally {
                        model.close()
                    }
                } else {
                    val model = NativeModel.load(checkpoint.absolutePath)
                        ?: throw IllegalStateException("Could not load the best checkpoint.")
                    try {
                        val trimmed = prompt.takeLast((model.config.contextLength - 1).coerceAtLeast(1))
                            .toIntArray()
                        _playground.update {
                            it.copy(status = "Prefilling ${trimmed.size} prompt tokens…")
                        }
                        val t0 = System.nanoTime()
                        started = t0
                        generated = model.generateStreaming(
                            trimmed, maxNew, state.temperature, state.topK, state.topP,
                            System.nanoTime(), onToken
                        )
                        elapsed = System.nanoTime() - t0
                    } finally {
                        model.close()
                    }
                }

                val seconds = elapsed / 1_000_000_000.0
                val text = tokenizer.decode(generated)
                val wasCancelled = playgroundCancel
                _playground.update {
                    it.copy(
                        busy = false,
                        status = if (wasCancelled) {
                            "Stopped after ${generated.size} tokens"
                        } else {
                            "${generated.size} tokens in " +
                                String.format("%.1f s", seconds)
                        },
                        output = text,
                        generatedTokens = generated.size,
                        tokensPerSec = if (seconds > 0.0) generated.size / seconds else 0.0,
                        latencyMs = elapsed / 1_000_000L,
                        cancelled = wasCancelled
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

    /** Stops the current generation between tokens and keeps the partial text. */
    fun cancelPlayground() {
        if (!_playground.value.busy) return
        playgroundCancel = true
        _playground.update { it.copy(status = "Stopping…") }
    }

    fun exportPacked() {
        val state = _playground.value
        if (state.busy || state.modelId.isBlank()) return
        val id = state.modelId
        val bits = state.bits
        if (bits !in intArrayOf(2, 3, 4, 8)) {
            _playground.update {
                it.copy(error = "Packed export supports 2, 3, 4 and 8 bit only.")
            }
            return
        }
        val checkpoint = inferenceCheckpoint(id)
        if (!CheckpointHeader.hasMagic(checkpoint)) {
            _playground.update {
                it.copy(error = "No checkpoint yet. Start or finish training first.")
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
                            exportedBytes = target.length(),
                            lastExportFile = target.absolutePath
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

    /** GGML tensor type: 0 = F32, 1 = F16, 2 = Q4_0, 8 = Q8_0. */
    fun exportGguf(ggmlType: Int) {
        if (ggmlType !in intArrayOf(0, 1, 2, 8)) return
        val label = when (ggmlType) {
            1 -> "GGUF F16"
            2 -> "GGUF Q4_0"
            8 -> "GGUF Q8_0"
            else -> "GGUF F32"
        }
        val suffix = when (ggmlType) {
            1 -> "f16"
            2 -> "q4_0"
            8 -> "q8_0"
            else -> "f32"
        }
        exportInterchange(label, "model-$suffix.gguf") { model, target ->
            model.saveGguf(target.absolutePath, ggmlType)
        }
    }

    fun exportSafetensors(fp16: Boolean) {
        exportInterchange(
            if (fp16) "safetensors F16" else "safetensors F32",
            if (fp16) "model-f16.safetensors" else "model-f32.safetensors"
        ) { model, target ->
            model.saveSafetensors(target.absolutePath, fp16)
        }
    }

    private fun exportInterchange(
        label: String,
        fileName: String,
        save: (NativeModel, File) -> Boolean
    ) {
        val state = _playground.value
        if (state.busy || state.modelId.isBlank()) return
        val id = state.modelId
        val checkpoint = inferenceCheckpoint(id)
        if (!CheckpointHeader.hasMagic(checkpoint)) {
            _playground.update {
                it.copy(error = "No checkpoint yet. Start or finish training first.")
            }
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            _playground.update {
                it.copy(busy = true, status = "Exporting $label…", error = null)
            }
            try {
                val model = NativeModel.load(checkpoint.absolutePath)
                    ?: throw IllegalStateException("Could not load the best checkpoint.")
                try {
                    val dir = File(store.modelDir(id), "export").apply { mkdirs() }
                    val target = File(dir, fileName)
                    if (!save(model, target)) {
                        throw IllegalStateException("The native exporter rejected the $label export.")
                    }
                    _playground.update {
                        it.copy(
                            busy = false,
                            status = "Exported ${target.name} (${formatBytes(target.length())})",
                            exportedBytes = target.length(),
                            lastExportFile = target.absolutePath
                        )
                    }
                } finally {
                    model.close()
                }
            } catch (t: Throwable) {
                _playground.update {
                    it.copy(busy = false, status = "", error = t.message ?: "$label export failed.")
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
        val saved = tokenStore.saveToken(value)
        _tokenPresent.value = tokenStore.token != null
        ForgeLog.info("credentials", "Hugging Face token saved=${tokenStore.token != null} ok=$saved")
        _notice.value = when {
            !saved -> "The token could not be saved securely. Try again or restart the app."
            value.isBlank() -> "Hugging Face token cleared."
            else -> "Token saved securely."
        }
    }

    fun clearToken() {
        val cleared = tokenStore.clear()
        _tokenPresent.value = tokenStore.token != null
        ForgeLog.info("credentials", "Hugging Face token cleared ok=$cleared")
        _notice.value = if (cleared) {
            "Hugging Face token cleared."
        } else {
            "The token could not be cleared from secure storage. Try again."
        }
    }

    fun clearNotice() {
        _notice.value = null
    }

    private fun freeDownloadBytes(modelId: String): Long {
        return try {
            val stat = StatFs(store.modelDir(modelId).path)
            (stat.availableBytes - 64L * 1024L * 1024L).coerceAtLeast(1_000_000L)
        } catch (_: Throwable) {
            Long.MAX_VALUE / 4L
        }
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
