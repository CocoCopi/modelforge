package com.modelforge.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ArchitectureRegistry
import com.modelforge.app.core.ArchitectureStatus
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.core.Modality
import com.modelforge.app.data.DatasetMixEntry
import com.modelforge.app.data.ai.AiDatasetSuggestion
import com.modelforge.app.data.hf.HfDatasetSummary
import com.modelforge.app.data.hf.isLikelyNonCommercial
import com.modelforge.app.ui.design.BitGrainSelector
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.design.Hairline
import com.modelforge.app.ui.design.LedgerStat
import com.modelforge.app.ui.design.NotebookCard
import com.modelforge.app.ui.design.ParameterRuler
import com.modelforge.app.ui.design.SectionHeading
import com.modelforge.app.ui.design.SmallCaps
import com.modelforge.app.ui.design.StatusTag
import com.modelforge.app.ui.design.WizardRuler
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType
import kotlinx.coroutines.delay

private val languageChoices = listOf(
    "English", "Spanish", "French", "German", "Portuguese",
    "Italian", "Hindi", "Japanese", "Chinese", "Arabic", "Tamil"
)

@Composable
fun WizardScreen(vm: ModelForgeViewModel) {
    val state by vm.wizard.collectAsState()
    val step = WizardStep.at(state.step)

    ForgeScaffold(
        title = "New model",
        subtitle = "Step ${state.step + 1} of ${WizardStep.entries.size} · ${step.label}",
        onBack = vm::wizardBack
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            WizardRuler(step = state.step, total = WizardStep.entries.size)
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                SmallCaps(step.label)
                Text(
                    step.question,
                    style = MaterialTheme.typography.displaySmall,
                    color = ForgeColors.ink
                )
                Text(
                    step.help,
                    style = MaterialTheme.typography.bodyMedium,
                    color = ForgeColors.inkMuted
                )
            }
            Hairline()

            when (step) {
                WizardStep.IDENTITY -> IdentityStep(vm, state)
                WizardStep.SIZE -> SizeStep(vm, state)
                WizardStep.ARCHITECTURE -> ArchitectureStep(vm, state)
                WizardStep.MODALITIES -> ModalitiesStep(vm, state)
                WizardStep.TRAINING_TIME -> TrainingTimeStep(vm, state)
                WizardStep.LANGUAGES -> LanguagesStep(vm, state)
                WizardStep.CONTEXT -> ContextStep(vm, state)
                WizardStep.SPEED -> SpeedStep(vm, state)
                WizardStep.REASONING -> ReasoningStep(vm, state)
                WizardStep.TOOLS -> ToolsStep(vm, state)
                WizardStep.PERSONA -> PersonaStep(vm, state)
                WizardStep.COMPRESSION -> CompressionStep(vm, state)
                WizardStep.AI_PLAN -> AiPlanStep(vm, state)
                WizardStep.DATASETS -> DatasetStep(vm, state)
                WizardStep.REVIEW -> ReviewStep(vm, state)
            }

            state.error?.let { message ->
                Text(message, color = ForgeColors.error, style = MaterialTheme.typography.bodySmall)
            }

            DraftNote()
            WizardNav(vm, state)
        }
    }
}

@Composable
private fun DraftNote() {
    Text(
        "Draft saved on this device.",
        style = ForgeType.monoSmall,
        color = ForgeColors.inkMuted
    )
}

@Composable
private fun IdentityStep(vm: ModelForgeViewModel, state: WizardState) {
    SectionCard("Identity") {
        OutlinedTextField(
            value = state.name,
            onValueChange = vm::updateName,
            label = { Text("Model name") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = state.creator,
            onValueChange = vm::updateCreator,
            label = { Text("Created by") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Hairline()
        SmallCaps("LIVE PREVIEW")
        Text(
            "Hi, I'm ${state.name.ifBlank { "your model" }}, created by ${state.creator.ifBlank { "you" }}.",
            style = MaterialTheme.typography.headlineSmall,
            color = ForgeColors.ink
        )
        Text(
            "The creator line is trained into the model. It is not a claim about another company's model.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun SizeStep(vm: ModelForgeViewModel, state: WizardState) {
    val config = state.config
    NotebookCard {
        ParameterRuler(
            value = state.targetParams,
            min = 1_000_000L,
            max = vm.maxTargetParams.coerceAtLeast(1_000_000L),
            onValueChange = vm::updateTarget
        )
        if (config != null) {
            Hairline()
            LedgerStat("counted", ConfigDeriver.formatCount(config.paramCount()))
            LedgerStat("layers", config.layers.toString())
            LedgerStat("width", config.dModel.toString())
            LedgerStat("heads", "${config.heads} × ${config.headDim}")
            LedgerStat("feed-forward", config.dFf.toString())
            LedgerStat("training memory", formatBytes(config.trainingBytes()))
            Text(
                "Device limit: ${ConfigDeriver.formatCount(vm.maxTargetParams)} parameters. " +
                    "Counted size is what training will allocate, not a marketing number.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun ArchitectureStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        ArchitectureRegistry.all().forEach { info ->
            val selected = info.id == state.architectureId
            NotebookCard(
                modifier = Modifier.clickable(enabled = info.available) {
                    vm.updateArchitecture(info.id)
                },
                selected = selected
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(info.displayName, style = MaterialTheme.typography.headlineSmall, color = ForgeColors.ink)
                    StatusTag(
                        info.status.label,
                        when (info.status) {
                            ArchitectureStatus.STABLE -> ForgeColors.verdigris
                            ArchitectureStatus.EXPERIMENTAL -> ForgeColors.ochreInk
                            ArchitectureStatus.PLANNED -> ForgeColors.inkMuted
                        }
                    )
                }
                Text(info.summary, style = MaterialTheme.typography.bodySmall, color = ForgeColors.inkMuted)
                Text(
                    "${info.inputs.joinToString(" + ") { it.label }} → " +
                        "${info.outputs.joinToString(" + ") { it.label }} · ${info.datasetFormat.label}",
                    style = ForgeType.monoSmall,
                    color = ForgeColors.inkMuted
                )
                info.unavailableReason?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = ForgeColors.ochreInk)
                }
            }
        }
    }
}

@Composable
private fun ModalitiesStep(vm: ModelForgeViewModel, state: WizardState) {
    val info = ArchitectureRegistry.info(state.architectureId)
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        SectionCard("Input") {
            ModalityChoices(state.inputModalities) { vm.toggleInputModality(it) }
        }
        SectionCard("Output") {
            ModalityChoices(state.outputModalities) { vm.toggleOutputModality(it) }
        }
        NotebookCard {
            LedgerStat("architecture", info.displayName)
            LedgerStat("dataset format", info.datasetFormat.label)
            LedgerStat(
                "quantization",
                if (info.supportedBits.isEmpty()) "not available" else "${info.supportedBits.joinToString("/")}-bit"
            )
            Text(
                "The NVIDIA planner uses these choices to recommend data. The current native trainer can " +
                    "only train text-to-text, so other modality combinations stay plan-only.",
                style = MaterialTheme.typography.bodySmall,
                color = if (state.inputModalities != setOf(Modality.TEXT) ||
                    state.outputModalities != setOf(Modality.TEXT)
                ) ForgeColors.ochreInk else ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun ModalityChoices(selected: Set<Modality>, onToggle: (Modality) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Modality.values().toList().chunked(2).forEach { row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                row.forEach { modality ->
                    Choice(
                        text = modality.label,
                        selected = modality in selected,
                        onClick = { onToggle(modality) },
                        modifier = Modifier.weight(1f)
                    )
                }
                if (row.size == 1) Box(Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun TrainingTimeStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(30, 60, 120, 240, 480).forEach { minutes ->
                Choice(
                    text = if (minutes < 60) "${minutes}m" else "${minutes / 60}h",
                    selected = state.trainingMinutes == minutes,
                    onClick = { vm.updateTrainingMinutes(minutes) }
                )
            }
        }
        NotebookCard {
            LedgerStat("time available", formatMinutes(state.trainingMinutes))
            LedgerStat("current steps", state.steps.toString())
            Slider(
                value = state.trainingMinutes.toFloat(),
                onValueChange = { vm.updateTrainingMinutes(it.toInt()) },
                valueRange = 15f..480f,
                steps = 30
            )
            Text(
                "The AI architect turns this time budget into steps, batch size and mix weights. " +
                    "Actual speed depends on this phone and is measured only after training starts.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun AiPlanStep(vm: ModelForgeViewModel, state: WizardState) {
    val ai by vm.aiPlanner.collectAsState()
    val keyPresent by vm.aiKeyPresent.collectAsState()
    var showKey by remember { mutableStateOf(false) }
    var elapsed by remember { mutableStateOf(0L) }
    LaunchedEffect(ai.busy, ai.startedAt) {
        if (!ai.busy) {
            elapsed = 0L
        } else {
            while (true) {
                elapsed = (System.currentTimeMillis() - ai.startedAt) / 1000L
                delay(1000)
            }
        }
    }
    SectionCard("NVIDIA architect") {
        LedgerStat("api key", if (keyPresent) "saved securely" else "not set")
        Text(
            "Uses ${com.modelforge.app.data.ai.NvidiaClient.MODEL} through NVIDIA's hosted API. " +
                "Reasoning is set to none so the JSON plan starts promptly. The key is stored with " +
                "EncryptedSharedPreferences, sent only to NVIDIA, and never logged.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ForgeOutlineButton(
                if (keyPresent) "Replace key" else "Add key",
                onClick = { showKey = true },
                modifier = Modifier.weight(1f)
            )
            if (keyPresent) {
                ForgeOutlineButton(
                    "Clear",
                    onClick = vm::clearAiKey,
                    modifier = Modifier.weight(1f),
                    tint = ForgeColors.error
                )
            }
        }
        ForgeButton(
            when {
                ai.busy -> "Designing…"
                ai.plan != null -> "Redesign with AI"
                else -> "Ask AI to design"
            },
            onClick = vm::requestAiPlan,
            enabled = keyPresent && !ai.busy,
            modifier = Modifier.fillMaxWidth(),
            primary = true
        )
        ai.error?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = ForgeColors.error)
        }
        if (ai.busy) {
            Hairline()
            Text(
                "${ai.stage.ifBlank { "Contacting NVIDIA" }} · ${elapsed}s · ${ai.events} events",
                style = ForgeType.monoSmall,
                color = ForgeColors.verdigris
            )
            if (ai.streaming.isBlank()) {
                Text(
                    "The AI is choosing training settings and checking real Hugging Face datasets. " +
                        "Downloads are added to the mix automatically.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
            } else {
                SmallCaps("STREAMING")
                Text(
                    ai.streaming.takeLast(1200),
                    style = ForgeType.monoSmall,
                    color = ForgeColors.inkMuted
                )
            }
            ForgeOutlineButton(
                "Cancel request",
                onClick = vm::cancelAiPlan,
                modifier = Modifier.fillMaxWidth(),
                tint = ForgeColors.error
            )
        }
        if (!ai.busy && ai.stage.isNotBlank()) {
            Text(ai.stage, style = MaterialTheme.typography.bodySmall, color = ForgeColors.verdigris)
        }
    }
    ai.plan?.let { plan ->
        SectionCard("Overview") {
            Text(
                plan.overview.ifBlank { "The AI returned no overview." },
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeColors.ink
            )
            if (plan.analogy.isNotBlank()) {
                Text(
                    plan.analogy,
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.verdigris
                )
            }
        }
        SectionCard("Training settings") {
            LedgerStat("steps", plan.training.steps.toString())
            LedgerStat("batch", plan.training.batchSize.toString())
            LedgerStat("sequence", plan.training.seqLength.toString())
            LedgerStat("learning rate", String.format("%.1e", plan.training.learningRate))
            LedgerStat("warmup", "${plan.training.warmupSteps} steps")
            Text(
                "Applied automatically. You can still adjust them on the previous screens.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
        SectionCard("Datasets chosen and downloaded by the AI") {
            if (plan.datasets.isEmpty()) {
                Text(
                    "The AI recommended no datasets. Open the dataset step to choose one manually.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.ochreInk
                )
            }
            plan.datasets.forEach { suggestion ->
                val status = when {
                    suggestion.datasetId in ai.appliedDatasetIds -> "downloaded"
                    suggestion.datasetId in ai.skippedDatasetIds -> "not used"
                    ai.busy -> "checking"
                    else -> "selected"
                }
                AiDatasetRow(suggestion, status, ai.skippedDatasetReasons[suggestion.datasetId])
            }
            if (!ai.busy && ai.skippedDatasetReasons.isNotEmpty()) {
                Hairline()
                Text(
                    "Why some suggestions were skipped",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ForgeColors.ochreInk
                )
                ai.skippedDatasetReasons.entries.take(5).forEach { (id, reason) ->
                    Text(
                        "$id — $reason",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
            }
            ForgeOutlineButton(
                "Open dataset mix",
                onClick = vm::goToDatasets,
                enabled = !ai.busy,
                modifier = Modifier.fillMaxWidth()
            )
        }
        if (plan.warnings.isNotEmpty()) {
            SectionCard("Warnings") {
                plan.warnings.forEach {
                    Text("· $it", style = MaterialTheme.typography.bodySmall, color = ForgeColors.ochreInk)
                }
            }
        }
        if (plan.notes.isNotEmpty()) {
            SectionCard("Design notes") {
                plan.notes.forEach {
                    Text("· $it", style = MaterialTheme.typography.bodySmall, color = ForgeColors.inkMuted)
                }
            }
        }
    }
    if (showKey) {
        AiKeyDialog(
            keyPresent = keyPresent,
            onDismiss = { showKey = false },
            onSave = { vm.saveAiKey(it); showKey = false },
            onClear = { vm.clearAiKey(); showKey = false }
        )
    }
}

@Composable
private fun AiDatasetRow(suggestion: AiDatasetSuggestion, status: String, skipReason: String?) {
    val tint = when (status) {
        "downloaded" -> ForgeColors.verdigris
        "not used" -> ForgeColors.ochreInk
        "checking" -> ForgeColors.inkMuted
        else -> ForgeColors.verdigris
    }
    NotebookCard {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                suggestion.datasetId,
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeColors.verdigris
            )
            StatusTag(status, tint)
        }
        Text(
            "${suggestion.rows} rows · ${String.format("%.1f", suggestion.weight)}x · " +
                "license ${suggestion.license ?: "unknown"}",
            style = ForgeType.monoSmall,
            color = ForgeColors.inkMuted
        )
        if (!suggestion.commercialUse) {
            Text(
                "Commercial use is not confirmed. Verify the licence before shipping.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.ochreInk
            )
        }
        if (suggestion.reason.isNotBlank()) {
            Text(suggestion.reason, style = MaterialTheme.typography.bodySmall, color = ForgeColors.inkMuted)
        }
        if (!skipReason.isNullOrBlank()) {
            Text(skipReason, style = MaterialTheme.typography.bodySmall, color = ForgeColors.ochreInk)
        }
    }
}

@Composable
private fun LanguagesStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        languageChoices.chunked(2).forEach { row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                row.forEach { language ->
                    Choice(
                        text = language,
                        selected = language in state.languages,
                        onClick = { vm.toggleLanguage(language) },
                        modifier = Modifier.weight(1f)
                    )
                }
                if (row.size == 1) {
                    Box(Modifier.weight(1f))
                }
            }
        }
        Text(
            "Identity, reasoning and tool-call examples are generated in the selected languages where the local templates allow it.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun ContextStep(vm: ModelForgeViewModel, state: WizardState) {
    val options = listOf(128, 256, 512, 1024)
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { length ->
                Choice(
                    text = "$length",
                    selected = (state.contextChoice.takeIf { it > 0 } ?: state.config?.contextLength) == length,
                    onClick = { vm.updateContext(length) },
                    modifier = Modifier.weight(1f)
                )
            }
        }
        NotebookCard {
            val context = state.contextChoice.takeIf { it > 0 } ?: state.config?.contextLength ?: 0
            LedgerStat("tokens at once", context.toString())
            LedgerStat("attention cost", "grows with context² during training")
            Text(
                "Longer context is not automatically better for a tiny model. 128-256 tokens is a sensible start for short text.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun SpeedStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        ChoiceCard(
            title = "Instant",
            body = "Fewest generated tokens and no visible thinking. Good for quick checks.",
            selected = state.speedMode == "instant",
            onClick = { vm.updateSpeedMode("instant") }
        )
        ChoiceCard(
            title = "Normal",
            body = "Balanced. Reasoning is shown when enabled and the token budget is moderate.",
            selected = state.speedMode == "normal",
            onClick = { vm.updateSpeedMode("normal") }
        )
        ChoiceCard(
            title = "Extra time",
            body = "Allows a longer thinking block before the answer. Slower, measured on this device.",
            selected = state.speedMode == "extra",
            onClick = { vm.updateSpeedMode("extra") }
        )
        Text(
            "Speed is an inference setting. You can change it again in the Test screen after training.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun ReasoningStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Choice(
                text = "Off",
                selected = !state.reasoningEnabled,
                onClick = { vm.updateReasoning(false) },
                modifier = Modifier.weight(1f)
            )
            Choice(
                text = "On",
                selected = state.reasoningEnabled,
                onClick = { vm.updateReasoning(true) },
                modifier = Modifier.weight(1f)
            )
        }
        if (state.reasoningEnabled) {
            SmallCaps("REASONING ROOM")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(128, 256, 512, 1024).forEach { tokens ->
                    Choice(
                        text = "$tokens",
                        selected = state.reasoningRoom == tokens,
                        onClick = { vm.updateReasoningRoom(tokens) },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
            val params = state.config?.paramCount() ?: 0L
            if (params < 50_000_000L) {
                Text(
                    "Below about 50M parameters, reasoning examples often add little and can crowd out ordinary answers. " +
                        "The app will still mix them in if you continue.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.ochreInk
                )
            }
        }
        Text(
            "Training data includes  thinking …  response examples. Instant mode can skip the thinking block at inference.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun ToolsStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Choice(
                text = "Off",
                selected = !state.toolCalling,
                onClick = { vm.updateToolCalling(false) },
                modifier = Modifier.weight(1f)
            )
            Choice(
                text = "On",
                selected = state.toolCalling,
                onClick = { vm.updateToolCalling(true) },
                modifier = Modifier.weight(1f)
            )
        }
        NotebookCard {
            Text(
                "When enabled, the dataset mix includes simple tool-call examples: a request, a call, and a result.",
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeColors.ink
            )
            Text(
                "The app does not execute tools. It only trains the text pattern and shows the parsed call.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun PersonaStep(vm: ModelForgeViewModel, state: WizardState) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        OutlinedTextField(
            value = state.persona,
            onValueChange = vm::updatePersona,
            label = { Text("Behaviour note") },
            minLines = 4,
            maxLines = 8,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            "Short and concrete works best, for example: \"Answer briefly, prefer plain words, admit when the answer is not in the training data.\"",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
        Text(
            "${state.persona.length} / 600 characters",
            style = ForgeType.monoSmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun CompressionStep(vm: ModelForgeViewModel, state: WizardState) {
    val info = ArchitectureRegistry.info(state.architectureId)
    val config = state.config
    NotebookCard {
        if (config == null || info.supportedBits.isEmpty()) {
            Text("Quantization is not available for this architecture.", color = ForgeColors.ochreInk)
        } else {
            BitGrainSelector(
                bits = state.bits,
                sizes = info.supportedBits.associateWith { config.packedExportBytes(it) },
                onSelect = vm::updateBits
            )
            Hairline()
            LedgerStat("fp32 checkpoint", formatBytes(config.fp32Bytes()))
            LedgerStat("packed estimate", formatBytes(config.packedExportBytes(state.bits)))
            Text(
                "QAT is applied for 2-8 bit exports. The estimate matches the exact export layout; embeddings and norms stay fp32.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun DatasetStep(vm: ModelForgeViewModel, state: WizardState) {
    SectionCard("Find a dataset") {
        OutlinedTextField(
            value = state.searchQuery,
            onValueChange = vm::updateSearchQuery,
            label = { Text("Search Hugging Face datasets") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            ForgeButton("Search", onClick = vm::searchDatasets, enabled = !state.searching, primary = true)
            if (state.searching) {
                Text("searching…", style = ForgeType.mono, color = ForgeColors.inkMuted)
            }
        }
        if (state.searchResults.isNotEmpty()) {
            Text(
                "${state.searchResults.size} results",
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
            state.searchResults.take(10).forEach { summary ->
                DatasetResultRow(summary, selected = state.selectedDataset?.id == summary.id) {
                    vm.selectDataset(summary)
                }
            }
        }
    }

    if (state.selectedDataset != null) {
        SelectedDatasetCard(vm, state, state.selectedDataset)
    }

    SectionCard("Dataset mix") {
        AutoDatasetRow(
            title = "Identity",
            detail = "${state.name.ifBlank { "model" }} by ${state.creator.ifBlank { "you" }} · ${state.languages.joinToString(", ")}"
        )
        if (state.reasoningEnabled) {
            AutoDatasetRow("Reasoning", "thinking examples · ${state.reasoningRoom} reserved tokens")
        }
        if (state.toolCalling) {
            AutoDatasetRow("Tool calls", "request → call → result examples")
        }
        Hairline()
        if (state.datasets.isEmpty()) {
            Text(
                "No downloaded datasets yet. Add at least one real corpus before review.",
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeColors.inkMuted
            )
        } else {
            state.datasets.forEachIndexed { index, entry ->
                DatasetEntryRow(entry) { vm.removeDataset(index) }
            }
        }
    }
}

@Composable
private fun AutoDatasetRow(title: String, detail: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = ForgeColors.ink)
            Text(detail, style = ForgeType.monoSmall, color = ForgeColors.inkMuted)
        }
        StatusTag("auto", ForgeColors.verdigris)
    }
}

@Composable
private fun DatasetResultRow(summary: HfDatasetSummary, selected: Boolean, onClick: () -> Unit) {
    NotebookCard(
        modifier = Modifier.clickable(onClick = onClick),
        selected = selected
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(summary.id, style = MaterialTheme.typography.bodyMedium, color = ForgeColors.verdigris)
            Text(
                "♥ ${summary.likes}",
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
        }
        Text(
            "${ConfigDeriver.formatCount(summary.downloads)} downloads · " +
                "license ${summary.resolvedLicense() ?: "unknown"}",
            style = ForgeType.monoSmall,
            color = ForgeColors.inkMuted
        )
    }
}

@Composable
private fun SelectedDatasetCard(
    vm: ModelForgeViewModel,
    state: WizardState,
    summary: HfDatasetSummary
) {
    val license = summary.resolvedLicense()
    SectionCard(summary.id) {
        Text(
            "License: ${license ?: "unknown"}",
            style = MaterialTheme.typography.bodySmall,
            color = if (isLikelyNonCommercial(license)) ForgeColors.ochreInk else ForgeColors.inkMuted
        )
        if (isLikelyNonCommercial(license)) {
            Text(
                "This license may be non-commercial or unknown. Check the dataset card before using the model commercially.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.ochreInk
            )
        }

        val structure = state.structure
        if (structure == null) {
            if (state.datasetBusy) {
                Text("inspecting dataset…", style = ForgeType.mono, color = ForgeColors.inkMuted)
            }
        } else {
            val configNames = structure.configs.map { it.name }
            val currentConfig = structure.configs.firstOrNull { it.name == state.configName }
            val splitNames = currentConfig?.splits.orEmpty()

            SmallCaps("CONFIG")
            DropdownSelector(
                label = "Choose config",
                options = configNames,
                selected = state.configName,
                modifier = Modifier.fillMaxWidth()
            ) { vm.selectConfig(it) }

            SmallCaps("SPLIT")
            DropdownSelector(
                label = "Choose split",
                options = splitNames,
                selected = state.splitName,
                modifier = Modifier.fillMaxWidth()
            ) { vm.selectSplit(it) }

            SmallCaps("TEXT COLUMNS")
            val columns = currentConfig?.textColumns.orEmpty()
            if (columns.isEmpty()) {
                Text(
                    "No string columns were detected for this config. Choose another config.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.ochreInk
                )
            } else {
                Row(
                    modifier = Modifier.horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    columns.forEach { column ->
                        Choice(
                            text = column,
                            selected = column in state.columns,
                            onClick = { vm.toggleColumn(column) }
                        )
                    }
                }
            }

            LedgerStat("rows", state.rows.toString())
            Slider(
                value = state.rows.toFloat(),
                onValueChange = { vm.updateRows(it.toInt()) },
                valueRange = 100f..5000f,
                steps = 48
            )
            LedgerStat("mix weight", String.format("%.1fx", state.weight))
            Slider(
                value = state.weight,
                onValueChange = vm::updateWeight,
                valueRange = 0.5f..5f,
                steps = 8
            )

            if (state.datasetBusy) {
                Text("downloading…", style = ForgeType.mono, color = ForgeColors.emberText)
                state.datasetProgress?.let { progress ->
                    Text(
                        "${progress.rows} rows · ${formatBytes(progress.bytes)}",
                        style = ForgeType.monoSmall,
                        color = ForgeColors.inkMuted
                    )
                }
            } else {
                ForgeButton(
                    "Download and add",
                    onClick = vm::downloadSelectedDataset,
                    enabled = columns.isNotEmpty() && state.columns.isNotEmpty() &&
                        state.configName.isNotBlank() && state.splitName.isNotBlank(),
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
            }
        }
    }
}

@Composable
private fun DatasetEntryRow(entry: DatasetMixEntry, onRemove: () -> Unit) {
    NotebookCard {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(entry.datasetId, style = MaterialTheme.typography.bodyMedium, color = ForgeColors.ink)
                Text(
                    "${entry.config}/${entry.split} · ${entry.rows} rows · " +
                        "${formatBytes(entry.bytes)} · ${String.format("%.1f", entry.weight)}x",
                    style = ForgeType.monoSmall,
                    color = ForgeColors.inkMuted
                )
                if (entry.nonCommercial) {
                    Text(
                        "Non-commercial or unknown license",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
            }
            ForgeOutlineButton("Remove", onClick = onRemove, tint = ForgeColors.error)
        }
    }
}

@Composable
private fun ReviewStep(vm: ModelForgeViewModel, state: WizardState) {
    val config = state.config
    val info = ArchitectureRegistry.info(state.architectureId)
    val presets by vm.presets.collectAsState()
    val ai by vm.aiPlanner.collectAsState()
    var presetName by remember { mutableStateOf("") }
    SectionCard("Review") {
        LedgerStat("name", state.name.ifBlank { "—" })
        LedgerStat("creator", state.creator.ifBlank { "—" })
        LedgerStat("architecture", info.displayName)
        LedgerStat("parameters", config?.let { ConfigDeriver.formatCount(it.paramCount()) } ?: "—")
        LedgerStat("inputs", state.inputModalities.sortedBy { it.ordinal }.joinToString(", ") { it.label })
        LedgerStat("outputs", state.outputModalities.sortedBy { it.ordinal }.joinToString(", ") { it.label })
        LedgerStat("training time", formatMinutes(state.trainingMinutes))
        LedgerStat("context", (state.contextChoice.takeIf { it > 0 } ?: config?.contextLength)?.toString() ?: "—")
        LedgerStat("languages", state.languages.joinToString(", ").ifBlank { "—" })
        LedgerStat("response speed", state.speedMode)
        LedgerStat("reasoning", if (state.reasoningEnabled) "${state.reasoningRoom} tokens" else "off")
        LedgerStat("tool calls", if (state.toolCalling) "on" else "off")
        LedgerStat("compression", "${state.bits}-bit")
        LedgerStat("steps", state.steps.toString())
        LedgerStat("datasets", state.datasets.size.toString())
        if (state.persona.isNotBlank()) {
            Hairline()
            SmallCaps("BEHAVIOUR")
            Text(state.persona, style = MaterialTheme.typography.bodyMedium, color = ForgeColors.ink)
        }
    }

    SectionCard("Datasets") {
        AutoDatasetRow("Identity", "${state.name.ifBlank { "model" }} by ${state.creator.ifBlank { "you" }}")
        if (state.reasoningEnabled) AutoDatasetRow("Reasoning", "${state.reasoningRoom} reserved tokens")
        if (state.toolCalling) AutoDatasetRow("Tool calls", "request → call → result")
        state.datasets.forEach { entry ->
            Text(
                "${entry.datasetId} · ${String.format("%.1f", entry.weight)}x · ${entry.rows} rows",
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
        }
    }

    if (ai.applied) {
        SectionCard("AI autopilot") {
            Text(
                ai.stage.ifBlank { "The AI chose the training recipe and downloaded datasets." },
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeColors.verdigris
            )
            LedgerStat("auto-downloaded", "${ai.appliedDatasetIds.size} datasets")
            if (ai.skippedDatasetIds.isNotEmpty()) {
                Text(
                    "Skipped: ${ai.skippedDatasetIds.joinToString(", ")}",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.ochreInk
                )
                ai.skippedDatasetReasons.entries.take(3).forEach { (id, reason) ->
                    Text(
                        "$id — $reason",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
            }
            ai.plan?.warnings?.take(3)?.forEach {
                Text("· $it", style = MaterialTheme.typography.bodySmall, color = ForgeColors.ochreInk)
            }
            ForgeOutlineButton(
                "Review AI design",
                onClick = vm::goToAiPlan,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }

    SectionCard("Presets") {
        Text(
            "Save these settings for another model. Datasets are not included; " +
                "you will still choose data for each model.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
        OutlinedTextField(
            value = presetName,
            onValueChange = { presetName = it.take(60) },
            label = { Text("Preset name") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        ForgeButton(
            "Save preset",
            onClick = {
                vm.savePreset(presetName)
                presetName = ""
            },
            enabled = presetName.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
            primary = true
        )
        presets.forEach { preset ->
            Hairline()
            Text(preset.name, style = MaterialTheme.typography.headlineSmall, color = ForgeColors.ink)
            Text(
                preset.summary,
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ForgeOutlineButton(
                    "Apply",
                    onClick = { vm.applyPreset(preset.id) },
                    modifier = Modifier.weight(1f)
                )
                ForgeOutlineButton(
                    "Delete",
                    onClick = { vm.deletePreset(preset.id) },
                    modifier = Modifier.weight(1f),
                    tint = ForgeColors.error
                )
            }
        }
    }

    SectionCard("Start") {
        Text(
            "Training runs on the phone in a foreground service. You can leave the app; use the notification to pause, resume or stop.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeColors.inkMuted
        )
        ForgeButton(
            "Start training",
            onClick = vm::startTraining,
            enabled = state.canStart,
            modifier = Modifier.fillMaxWidth(),
            primary = true
        )
        if (!state.canStart) {
            Text(
                when {
                    !info.trainable -> "This architecture is a planning path. Native training is not built yet."
                    state.inputModalities != setOf(Modality.TEXT) ||
                        state.outputModalities != setOf(Modality.TEXT) ->
                        "Native training currently supports text-to-text only."
                    state.datasets.isEmpty() -> "Add at least one dataset before starting."
                    state.name.isBlank() || state.creator.isBlank() ->
                        "Add a name and a creator before starting."
                    else -> "Check the model plan before starting."
                },
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.ochreInk
            )
        }
    }
}

@Composable
private fun ChoiceCard(
    title: String,
    body: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    NotebookCard(
        modifier = Modifier.clickable(onClick = onClick),
        selected = selected
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(title, style = MaterialTheme.typography.headlineSmall, color = ForgeColors.ink)
            if (selected) StatusTag("selected", ForgeColors.emberText)
        }
        Text(body, style = MaterialTheme.typography.bodySmall, color = ForgeColors.inkMuted)
    }
}

@Composable
private fun Choice(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .heightIn(min = 44.dp)
            .background(
                if (selected) ForgeColors.inkWash else ForgeColors.surface,
                RoundedCornerShape(4.dp)
            )
            .border(
                1.dp,
                if (selected) ForgeColors.emberText else ForgeColors.hairline,
                RoundedCornerShape(4.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text,
            style = MaterialTheme.typography.labelLarge,
            color = if (selected) ForgeColors.emberText else ForgeColors.ink
        )
    }
}

@Composable
private fun WizardNav(vm: ModelForgeViewModel, state: WizardState) {
    val step = WizardStep.at(state.step)
    val blocked = advanceBlockedReason(state)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (step != WizardStep.REVIEW) {
            ForgeButton(
                "Continue",
                onClick = vm::nextStep,
                enabled = blocked == null,
                modifier = Modifier.fillMaxWidth(),
                primary = true
            )
            if (blocked != null) {
                Text(blocked, style = MaterialTheme.typography.bodySmall, color = ForgeColors.ochreInk)
            }
        }
    }
}

private fun advanceBlockedReason(state: WizardState): String? {
    return when (WizardStep.at(state.step)) {
        WizardStep.IDENTITY -> when {
            state.name.isBlank() -> "Give the model a name."
            state.creator.isBlank() -> "Say who created it."
            else -> null
        }
        WizardStep.SIZE -> if (state.config == null &&
            ArchitectureRegistry.info(state.architectureId).trainable
        ) "Choose a size that fits this device." else null
        WizardStep.ARCHITECTURE -> if (!ArchitectureRegistry.info(state.architectureId).available) {
            "Choose an available architecture."
        } else null
        WizardStep.MODALITIES -> null
        WizardStep.TRAINING_TIME -> if (state.trainingMinutes < 15) "Choose at least 15 minutes." else null
        WizardStep.LANGUAGES -> if (state.languages.isEmpty()) "Choose at least one language." else null
        WizardStep.CONTEXT -> if ((state.contextChoice.takeIf { it > 0 } ?: state.config?.contextLength ?: 0) <= 0) {
            "Choose a context length."
        } else null
        WizardStep.SPEED -> null
        WizardStep.REASONING -> null
        WizardStep.TOOLS -> null
        WizardStep.PERSONA -> null
        WizardStep.COMPRESSION -> {
            val bits = ArchitectureRegistry.supportedBits(state.architectureId)
            if (bits.isNotEmpty() && state.bits !in bits) "Choose a supported compression." else null
        }
        WizardStep.AI_PLAN -> null
        WizardStep.DATASETS -> if (state.datasets.isEmpty()) "Add at least one dataset." else null
        WizardStep.REVIEW -> null
    }
}

private fun formatCount(value: Long): String = ConfigDeriver.formatCount(value)

private fun formatMinutes(minutes: Int): String =
    if (minutes < 60) "$minutes min" else "${minutes / 60} h ${minutes % 60} min".replace(" 0 min", "")
