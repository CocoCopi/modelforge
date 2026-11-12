package com.modelforge.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingStatus
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeBlue
import com.modelforge.app.ui.theme.ForgeDanger
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ForgeWarning
import kotlinx.coroutines.delay

@Composable
fun ModelDetailScreen(vm: ModelForgeViewModel, modelId: String) {
    val manifest = remember(modelId) { vm.manifest(modelId) }
    var state by remember(modelId) { mutableStateOf(vm.liveState(modelId)) }
    var confirmDelete by remember { mutableStateOf(false) }

    LaunchedEffect(modelId) {
        while (true) {
            state = vm.liveState(modelId)
            delay(1500)
        }
    }

    ForgeScaffold(
        title = manifest?.name ?: "Model",
        subtitle = "Model details",
        onBack = vm::goHome
    ) {
        if (manifest == null) {
            Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("This model was deleted or its manifest is missing.", color = ForgeDanger)
                OutlinedButton(onClick = vm::goHome) { Text("Back to models") }
            }
            return@ForgeScaffold
        }

        val config = manifest.config
        val checkpoints = vm.checkpointMegabytes(modelId)
        val estimatedPacked = config.packedExportBytes(config.bits)
        val status = state?.status ?: "not started"
        val active = state?.isActive == true

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            SectionCard("Architecture") {
                InfoRow("Parameters", formatParams(config.paramCount()), ForgeAccent)
                InfoRow("Layers", config.layers.toString())
                InfoRow("Width", "${config.dModel} (${config.heads} heads × ${config.headDim})")
                InfoRow("Feed-forward", config.dFf.toString())
                InfoRow("Context", config.contextLength.toString())
                InfoRow("Vocabulary", config.vocabSize.toString())
                InfoRow("Quantization", "${config.bits}-bit" + if (config.qat) " with QAT" else "")
                InfoRow("Target", formatParams(config.targetParams))
            }

            SectionCard("Training") {
                InfoRow("Status", statusLabel(status), statusColor(status))
                InfoRow(
                    "Step",
                    "${state?.step ?: 0} / ${state?.totalSteps?.takeIf { it > 0 } ?: manifest.settings.steps}"
                )
                InfoRow("Last loss", formatLoss(state?.loss ?: 0f))
                InfoRow("Best loss", formatLoss(state?.bestLoss ?: Float.MAX_VALUE))
                InfoRow("Speed", formatTokensPerSecond(state?.tokensPerSec ?: 0.0))
                InfoRow("Elapsed", formatDuration((state?.elapsedMs ?: 0L) / 1000L))
                if (state?.error != null) {
                    Text(state!!.error.orEmpty(), color = ForgeDanger)
                }
            }

            SectionCard("Datasets") {
                if (manifest.datasets.isEmpty()) {
                    Text("No datasets recorded.", color = ForgeMuted)
                } else {
                    manifest.datasets.forEach { entry ->
                        Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                            Text(entry.datasetId, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                "${entry.config}/${entry.split} • ${entry.rows} rows • " +
                                    "${String.format("%.1f", entry.weight)}x • " +
                                    "license ${entry.license ?: "unknown"}",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (entry.nonCommercial) ForgeWarning else ForgeMuted
                            )
                        }
                    }
                }
            }

            SectionCard("Storage") {
                InfoRow("Full-precision estimate", formatBytes(config.fp32Bytes()))
                InfoRow("Packed estimate (${config.bits}-bit)", formatBytes(estimatedPacked), ForgeAccent)
                InfoRow("Last checkpoint", if (checkpoints.first > 0) "${checkpoints.first} MB" else "none")
                InfoRow("Best checkpoint", if (checkpoints.second > 0) "${checkpoints.second} MB" else "none")
                Text(
                    "The packed estimate uses the exact export layout. Actual files are written " +
                        "when you export from the playground.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeMuted
                )
            }

            SectionCard("Sample") {
                val sample = state?.sample.orEmpty()
                Text(
                    sample.ifBlank {
                        "No sample yet. Samples are written every ${manifest.settings.sampleEvery} steps."
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    color = if (sample.isBlank()) ForgeMuted else MaterialTheme.colorScheme.onSurface
                )
            }

            SectionCard("Actions") {
                if (active) {
                    Button(onClick = { vm.openTraining(modelId) }, modifier = Modifier.fillMaxWidth()) {
                        Text("Open live training")
                    }
                } else {
                    Button(onClick = { vm.openTraining(modelId) }, modifier = Modifier.fillMaxWidth()) {
                        Text(
                            when (status) {
                                TrainingStatus.COMPLETED -> "Train again"
                                TrainingStatus.STOPPED, TrainingStatus.ERROR -> "Resume training"
                                else -> "Start training"
                            }
                        )
                    }
                }
                OutlinedButton(
                    onClick = { confirmDelete = true },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !active
                ) {
                    Text("Delete model", color = ForgeDanger)
                }
                if (active) {
                    Text(
                        "Stop training before deleting this model.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                }
                Text(
                    "Playground text generation and packed export arrive in the next update. " +
                        "This model's best checkpoint is ready.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeBlue
                )
            }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete ${manifest?.name ?: "model"}?") },
            text = {
                Text("This removes the model, datasets and all checkpoints from this device.")
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    vm.deleteModel(modelId)
                }) { Text("Delete", color = ForgeDanger) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancel") }
            }
        )
    }
}
