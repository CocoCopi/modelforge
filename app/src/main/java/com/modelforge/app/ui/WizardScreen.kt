package com.modelforge.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.data.DatasetMixEntry
import com.modelforge.app.data.hf.HfDatasetSummary
import com.modelforge.app.data.hf.isLikelyNonCommercial
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeBlue
import com.modelforge.app.ui.theme.ForgeDanger
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ForgeSurfaceHigh
import com.modelforge.app.ui.theme.ForgeWarning

private val stepLabels = listOf("Size", "Data", "Training", "Review")

@Composable
fun WizardScreen(vm: ModelForgeViewModel) {
    val state by vm.wizard.collectAsState()
    val targetOptions = vm.targetOptions()

    ForgeScaffold(
        title = "New model",
        subtitle = "Step ${state.step + 1} of 4 • ${stepLabels[state.step]}",
        onBack = vm::wizardBack
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            StepBar(state.step)
            when (state.step) {
                0 -> SizeStep(vm, state, targetOptions)
                1 -> DataStep(vm, state)
                2 -> TrainingStep(vm, state)
                else -> ReviewStep(vm, state)
            }
            state.error?.let { message ->
                Text(message, color = ForgeDanger, style = MaterialTheme.typography.bodyMedium)
            }
            WizardNav(vm, state)
        }
    }
}

@Composable
private fun StepBar(current: Int) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        stepLabels.forEachIndexed { index, label ->
            val active = index == current
            val done = index < current
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(22.dp)
                        .padding(1.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        if (done) "✓" else "${index + 1}",
                        color = when {
                            active -> ForgeAccent
                            done -> ForgeBlue
                            else -> ForgeMuted
                        },
                        fontWeight = if (active) FontWeight.Bold else FontWeight.Normal
                    )
                }
                Text(
                    label,
                    style = MaterialTheme.typography.labelSmall,
                    color = if (active) ForgeAccent else ForgeMuted
                )
            }
        }
    }
}

@Composable
private fun SizeStep(vm: ModelForgeViewModel, state: WizardState, targets: List<Long>) {
    SectionCard("Name and size") {
        OutlinedTextField(
            value = state.name,
            onValueChange = vm::updateName,
            label = { Text("Model name") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Text("Target parameters", style = MaterialTheme.typography.labelMedium, color = ForgeMuted)
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            targets.forEach { target ->
                FilterChip(
                    selected = state.targetParams == target,
                    onClick = { vm.updateTarget(target) },
                    label = { Text(ConfigDeriver.formatCount(target)) }
                )
            }
        }
        Text(
            "The device limit is ${ConfigDeriver.formatCount(vm.maxTargetParams)} parameters. " +
                "Larger models take much longer to train.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
    }

    val config = state.config
    if (config != null) {
        val derived = ConfigDeriver.derive(
            state.targetParams,
            ConfigDeriver.Options(maxParams = vm.maxTargetParams)
        )
        SectionCard("Architecture") {
            InfoRow("Counted parameters", ConfigDeriver.formatCount(config.paramCount()), ForgeAccent)
            InfoRow("Layers", config.layers.toString())
            InfoRow("Width", "${config.dModel} (${config.heads} heads × ${config.headDim})")
            InfoRow("Feed-forward", config.dFf.toString())
            InfoRow("Context", config.contextLength.toString())
            InfoRow("Vocabulary", config.vocabSize.toString())
            Text(
                derived.explanation + " Parameter count is within " +
                    String.format("%.2f%%", derived.errorFraction * 100.0) + " of the target.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeMuted
            )
        }
    }

    SectionCard("Quantization") {
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(4, 8, 32).forEach { bits ->
                FilterChip(
                    selected = state.bits == bits,
                    onClick = { vm.updateBits(bits) },
                    label = { Text("$bits-bit") }
                )
            }
        }
        Text(
            when (state.bits) {
                4 -> "4-bit is the recommended default: about 8x smaller linear weights with QAT."
                8 -> "8-bit is safer for quality and still about 4x smaller."
                else -> "32-bit keeps full precision and the largest files; no QAT is used."
            },
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
    }
}

@Composable
private fun DataStep(vm: ModelForgeViewModel, state: WizardState) {
    SectionCard("Find a dataset") {
        OutlinedTextField(
            value = state.searchQuery,
            onValueChange = vm::updateSearchQuery,
            label = { Text("Search Hugging Face datasets") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = vm::searchDatasets, enabled = !state.searching) {
                Icon(Icons.Filled.Search, contentDescription = null)
                Text("  Search")
            }
            if (state.searching) {
                CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.dp)
            }
        }
        if (state.searchResults.isNotEmpty()) {
            Text(
                "${state.searchResults.size} results",
                style = MaterialTheme.typography.labelSmall,
                color = ForgeMuted
            )
            state.searchResults.take(12).forEach { summary ->
                DatasetResultRow(summary, selected = state.selectedDataset?.id == summary.id) {
                    vm.selectDataset(summary)
                }
            }
        }
    }

    if (state.selectedDataset != null) {
        SelectedDatasetCard(vm, state, state.selectedDataset)
    }

    SectionCard("Selected datasets") {
        if (state.datasets.isEmpty()) {
            Text(
                "No datasets yet. Search above, choose text columns, and download at least one dataset.",
                style = MaterialTheme.typography.bodyMedium,
                color = ForgeMuted
            )
        } else {
            state.datasets.forEachIndexed { index, entry ->
                DatasetEntryRow(entry) { vm.removeDataset(index) }
            }
        }
    }
}

@Composable
private fun DatasetResultRow(summary: HfDatasetSummary, selected: Boolean, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) ForgeSurfaceHigh else MaterialTheme.colorScheme.surface
        )
    ) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(summary.id, style = MaterialTheme.typography.bodyMedium, color = ForgeBlue)
            Text(
                "${ConfigDeriver.formatCount(summary.downloads)} downloads • " +
                    "license ${summary.resolvedLicense() ?: "unknown"} • ♥ ${summary.likes}",
                style = MaterialTheme.typography.labelSmall,
                color = ForgeMuted
            )
        }
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
            color = if (isLikelyNonCommercial(license)) ForgeWarning else ForgeMuted
        )
        if (isLikelyNonCommercial(license)) {
            Text(
                "This license may be non-commercial or unknown. Check the dataset card before " +
                    "using the trained model commercially.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeWarning
            )
        }

        val structure = state.structure
        if (structure == null) {
            if (state.datasetBusy) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Text("  Inspecting dataset…", style = MaterialTheme.typography.bodySmall)
                }
            }
        } else {
            val configNames = structure.configs.map { it.name }
            val currentConfig = structure.configs.firstOrNull { it.name == state.configName }
            val splitNames = currentConfig?.splits.orEmpty()

            Text("Config", style = MaterialTheme.typography.labelMedium, color = ForgeMuted)
            DropdownSelector(
                label = "Choose config",
                options = configNames,
                selected = state.configName,
                modifier = Modifier.fillMaxWidth()
            ) { vm.selectConfig(it) }

            Text("Split", style = MaterialTheme.typography.labelMedium, color = ForgeMuted)
            DropdownSelector(
                label = "Choose split",
                options = splitNames,
                selected = state.splitName,
                modifier = Modifier.fillMaxWidth()
            ) { vm.selectSplit(it) }

            Text("Text columns", style = MaterialTheme.typography.labelMedium, color = ForgeMuted)
            val columns = currentConfig?.textColumns.orEmpty()
            if (columns.isEmpty()) {
                Text(
                    "No string columns were detected for this config. Choose another config.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeWarning
                )
            } else {
                Row(
                    modifier = Modifier.horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    columns.forEach { column ->
                        FilterChip(
                            selected = column in state.columns,
                            onClick = { vm.toggleColumn(column) },
                            label = { Text(column) }
                        )
                    }
                }
            }

            Text(
                "Rows: ${state.rows}",
                style = MaterialTheme.typography.labelMedium,
                color = ForgeMuted
            )
            Slider(
                value = state.rows.toFloat(),
                onValueChange = { vm.updateRows(it.toInt()) },
                valueRange = 100f..5000f,
                steps = 48
            )

            Text(
                "Weight: ${String.format("%.1f", state.weight)}x",
                style = MaterialTheme.typography.labelMedium,
                color = ForgeMuted
            )
            Slider(
                value = state.weight,
                onValueChange = vm::updateWeight,
                valueRange = 0.5f..5f,
                steps = 8
            )

            if (state.datasetBusy) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                val progress = state.datasetProgress
                if (progress != null) {
                    Text(
                        "Downloaded ${progress.rows} rows • ${formatBytes(progress.bytes)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                }
            } else {
                Button(
                    onClick = vm::downloadSelectedDataset,
                    enabled = columns.isNotEmpty() && state.columns.isNotEmpty() &&
                        state.configName.isNotBlank() && state.splitName.isNotBlank(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Download and add dataset")
                }
            }
        }
    }
}

@Composable
private fun DatasetEntryRow(entry: DatasetMixEntry, onRemove: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = ForgeSurfaceHigh)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(entry.datasetId, style = MaterialTheme.typography.bodyMedium)
                Text(
                    "${entry.config}/${entry.split} • ${entry.rows} rows • " +
                        "${formatBytes(entry.bytes)} • ${String.format("%.1f", entry.weight)}x",
                    style = MaterialTheme.typography.labelSmall,
                    color = ForgeMuted
                )
                if (entry.nonCommercial) {
                    Text(
                        "Non-commercial or unknown license",
                        style = MaterialTheme.typography.labelSmall,
                        color = ForgeWarning
                    )
                }
            }
            TextButton(onClick = onRemove) {
                Icon(Icons.Filled.Delete, contentDescription = "Remove dataset", tint = ForgeDanger)
            }
        }
    }
}

@Composable
private fun TrainingStep(vm: ModelForgeViewModel, state: WizardState) {
    SectionCard("Training plan") {
        Text(
            "Steps: ${state.steps}",
            style = MaterialTheme.typography.labelMedium,
            color = ForgeMuted
        )
        Slider(
            value = state.steps.toFloat(),
            onValueChange = { vm.updateSteps(it.toInt()) },
            valueRange = 100f..2000f,
            steps = 18
        )
        InfoRow("Batch size", state.batchSize.toString())
        InfoRow("Sequence length", state.seqLength.toString())
        InfoRow("Learning rate", String.format("%.1e", state.learningRate))
        InfoRow("Warmup", "${state.warmupSteps} steps")
        InfoRow("Checkpoint every", "${state.checkpointEvery} steps")
        InfoRow("Sample every", "${state.sampleEvery} steps")
        InfoRow("Tokenizer corpus", "${ConfigDeriver.formatCount(state.corpusChars.toLong())} chars")
        Text(
            "Batch, sequence length and learning rate are chosen from the model size and " +
                "available RAM. Training measures real tokens/second so the ETA is not a guess.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
    }

    SectionCard("Device safety") {
        Text(
            "Training runs in a foreground service. It pauses automatically below 15% battery " +
                "or when the phone runs hot, and resumes when conditions are safe. A partial " +
                "wake lock is held only while training.",
            style = MaterialTheme.typography.bodyMedium,
            color = ForgeMuted
        )
        InfoRow("Quantization", "${state.bits}-bit")
        InfoRow("Auto-resume", "enabled")
    }
}

@Composable
private fun ReviewStep(vm: ModelForgeViewModel, state: WizardState) {
    val config = state.config
    SectionCard("Review") {
        InfoRow("Name", state.name.ifBlank { "My model" })
        InfoRow("Target", ConfigDeriver.formatCount(state.targetParams))
        InfoRow("Counted parameters", config?.let { ConfigDeriver.formatCount(it.paramCount()) } ?: "—", ForgeAccent)
        InfoRow("Layers", config?.layers?.toString() ?: "—")
        InfoRow("Width", config?.dModel?.toString() ?: "—")
        InfoRow("Context", config?.contextLength?.toString() ?: "—")
        InfoRow("Vocabulary", config?.vocabSize?.toString() ?: "—")
        InfoRow("Quantization", "${state.bits}-bit")
        InfoRow("Steps", state.steps.toString())
        InfoRow("Datasets", state.datasets.size.toString())
    }

    SectionCard("Datasets") {
        if (state.datasets.isEmpty()) {
            Text("Add at least one dataset in the data step.", color = ForgeWarning)
        } else {
            state.datasets.forEach { entry ->
                Column(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    Text(entry.datasetId, style = MaterialTheme.typography.bodyMedium)
                    Text(
                        "${entry.config}/${entry.split} • ${entry.rows} rows • " +
                            "${String.format("%.1f", entry.weight)}x",
                        style = MaterialTheme.typography.labelSmall,
                        color = ForgeMuted
                    )
                }
            }
        }
    }

    SectionCard("Start") {
        Text(
            "Training runs on the phone. You can leave the app; use the notification to pause, " +
                "resume or stop. The best checkpoint is kept for export in the playground.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
        Button(
            onClick = vm::startTraining,
            enabled = state.canStart,
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(Icons.Filled.Check, contentDescription = null)
            Text("  Start training")
        }
        if (!state.canStart) {
            Text(
                "Add a name and at least one dataset before starting.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeWarning
            )
        }
    }
}

@Composable
private fun WizardNav(vm: ModelForgeViewModel, state: WizardState) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        if (state.step > 0) {
            OutlinedButton(onClick = vm::previousStep, modifier = Modifier.weight(1f)) {
                Text("Back")
            }
        }
        if (state.step < 3) {
            val enabled = when (state.step) {
                0 -> state.name.isNotBlank() && state.config != null
                1 -> state.datasets.isNotEmpty()
                else -> true
            }
            Button(
                onClick = vm::nextStep,
                enabled = enabled,
                modifier = Modifier.weight(1f)
            ) {
                Text("Continue")
            }
        }
    }
    if (state.step == 1 && state.datasets.isEmpty()) {
        Text(
            "Download at least one dataset to continue.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
    }
}
