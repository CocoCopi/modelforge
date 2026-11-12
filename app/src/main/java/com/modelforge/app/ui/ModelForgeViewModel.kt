package com.modelforge.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.modelforge.app.core.ArchitectureRegistry
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.core.ModelConfig
import com.modelforge.app.data.DatasetMixEntry
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.SecureTokenStore
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingSettings
import com.modelforge.app.data.WizardDraft
import com.modelforge.app.data.WizardDraftStore
import com.modelforge.app.data.hf.DatasetStructure
import com.modelforge.app.data.hf.DownloadStats
import com.modelforge.app.data.hf.HfClient
import com.modelforge.app.data.hf.HfDatasetSummary
import com.modelforge.app.data.hf.HfRepository
import com.modelforge.app.data.hf.isLikelyNonCommercial
import com.modelforge.app.data.tokenizer.BpeTokenizer
import com.modelforge.app.nativebridge.NativeModel
import com.modelforge.app.nativebridge.PackedModel
import com.modelforge.app.training.TrainingService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

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
            ArchitectureRegistry.info(architectureId).available
}

data class PlaygroundState(
    val modelId: String = "",
    val prompt: String = "Once upon a time",
    val maxNew: Int = 120,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val mode: String = "checkpoint",
    val bits: Int = 4,
    val output: String = "",
    val busy: Boolean = false,
    val status: String = "",
    val tokensPerSec: Double = 0.0,
    val generatedTokens: Int = 0,
    val exportedBytes: Long = 0L,
    val error: String? = null
) {
    companion object {
        const val MODE_CHECKPOINT = "checkpoint"
        const val MODE_PACKED = "packed"
    }
}

class ModelForgeViewModel(app: Application) : AndroidViewModel(app) {

    private val store = ModelStore(app)
    private val draftStore = WizardDraftStore(app)
    private val tokenStore = SecureTokenStore(app)
    private val hf = HfRepository(HfClient.create { tokenStore.token })

    val hardware = HardwareDetector.detect(app)
    val maxTargetParams: Long = hardware.maxTrainableParams

    private val _screen = MutableStateFlow<Screen>(Screen.Home)
    val screen: StateFlow<Screen> = _screen.asStateFlow()

    private val _models = MutableStateFlow(store.listManifests())
    val models: StateFlow<List<ModelManifest>> = _models.asStateFlow()

    private val _wizard = MutableStateFlow(WizardState())
    val wizard: StateFlow<WizardState> = _wizard.asStateFlow()

    private val _tokenPresent = MutableStateFlow(tokenStore.token != null)
    val tokenPresent: StateFlow<Boolean> = _tokenPresent.asStateFlow()

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
            val entry = state.datasets.getOrNull(index) ?: return@update state
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

    fun pauseTraining() = TrainingService.pause(getApplication())

    fun resumeTraining() = TrainingService.resume(getApplication())

    fun stopTraining() = TrainingService.stop(getApplication())

    fun openPlayground(id: String) {
        val manifest = store.loadManifest(id) ?: return
        val bits = manifest.config.bits.takeIf { it in 2..8 } ?: 4
        _playground.value = PlaygroundState(modelId = id, bits = bits)
        _screen.value = Screen.Playground(id)
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
                val encoded = tokenizer.encode(state.prompt)
                val prompt = if (encoded.isEmpty()) tokenizer.encode(" ") else encoded
                val maxNew = state.maxNew.coerceIn(1, 512)
                var text = ""
                var tokens = 0
                var perSecond = 0.0
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
                        val seconds = (System.nanoTime() - started) / 1_000_000_000.0
                        text = tokenizer.decode(trimmed + generated)
                        tokens = generated.size
                        perSecond = if (seconds > 0.0) generated.size / seconds else 0.0
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
                        val seconds = (System.nanoTime() - started) / 1_000_000_000.0
                        text = tokenizer.decode(trimmed + generated)
                        tokens = generated.size
                        perSecond = if (seconds > 0.0) generated.size / seconds else 0.0
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
                        tokensPerSec = perSecond
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
