package com.modelforge.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingStatus
import kotlinx.coroutines.delay
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

@Composable
fun ModelDetailScreen(vm: ModelForgeViewModel, modelId: String) {
    val manifest = remember(modelId) { vm.manifest(modelId) }
    var state by remember(modelId) { mutableStateOf(vm.liveState(modelId)) }
    var confirmDelete by remember { mutableStateOf(false) }
    val identity by vm.identityCheck.collectAsState()

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
                Text("This model was deleted or its manifest is missing.", color = ForgeColors.error)
                ForgeOutlineButton("Back to models", onClick = vm::goHome)
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
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            SectionCard("Architecture") {
                InfoRow("Parameters", formatParams(config.paramCount()), ForgeColors.emberText)
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
                    Text(state!!.error.orEmpty(), color = ForgeColors.error)
                }
            }

            SectionCard("Datasets") {
                if (manifest.datasets.isEmpty()) {
                    Text("No datasets recorded.", color = ForgeColors.inkMuted)
                } else {
                    manifest.datasets.forEach { entry ->
                        Column(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                            Text(entry.datasetId, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                "${entry.config}/${entry.split} • ${entry.rows} rows • " +
                                    "${String.format("%.1f", entry.weight)}x • " +
                                    "license ${entry.license ?: "unknown"}",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (entry.nonCommercial) ForgeColors.ochreInk else ForgeColors.inkMuted
                            )
                        }
                    }
                }
            }

            SectionCard("Storage") {
                InfoRow("Full-precision estimate", formatBytes(config.fp32Bytes()))
                InfoRow("Packed estimate (${config.bits}-bit)", formatBytes(estimatedPacked), ForgeColors.emberText)
                InfoRow("Last checkpoint", if (checkpoints.first > 0) "${checkpoints.first} MB" else "none")
                InfoRow("Best checkpoint", if (checkpoints.second > 0) "${checkpoints.second} MB" else "none")
                Text(
                    "The packed estimate uses the exact export layout. Actual files are written " +
                        "when you export from the playground.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
            }

            SectionCard("Sample") {
                val sample = state?.sample.orEmpty()
                Text(
                    sample.ifBlank {
                        "No sample yet. Samples are written every ${manifest.settings.sampleEvery} steps."
                    },
                    style = ForgeType.mono,
                    color = if (sample.isBlank()) ForgeColors.inkMuted else ForgeColors.ink
                )
            }

            val identityHere = identity.modelId == modelId
            SectionCard("Identity check") {
                Text(
                    "Ask 20 identity questions and measure how often the model names " +
                        "itself and its creator. Runs locally on the best checkpoint.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
                ForgeButton(
                    if (identityHere && identity.busy) {
                        "Checking ${identity.total}/20…"
                    } else {
                        "Run 20-question identity check"
                    },
                    onClick = { vm.checkIdentity(modelId) },
                    enabled = !active && !(identityHere && identity.busy) &&
                        checkpoints.second > 0,
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
                if (checkpoints.second == 0) {
                    Text(
                        "Train this model first; there is no best checkpoint to test.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
                if (identityHere && identity.error != null) {
                    Text(
                        identity.error.orEmpty(),
                        color = ForgeColors.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                if (identityHere && identity.total > 0) {
                    InfoRow(
                        "Score",
                        "${identity.passed} / ${identity.total} " +
                            "(${String.format("%.0f", identity.score * 100)}%)",
                        if (identity.needsReinforce) ForgeColors.ochreInk else ForgeColors.verdigris
                    )
                }
                if (identityHere && identity.needsReinforce) {
                    Text(
                        "Below 80%. One-tap reinforce adds 100 steps with at least 8% " +
                            "identity data, resuming from the current checkpoint.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                    ForgeOutlineButton(
                        "Reinforce identity",
                        onClick = { vm.reinforceIdentity(modelId) },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !active
                    )
                }
                if (identityHere && identity.completed) {
                    identity.answers.filter { !it.passed }.take(3).forEach { answer ->
                        Column(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                            Text(answer.question, style = MaterialTheme.typography.bodySmall)
                            Text(
                                answer.response.ifBlank { "(empty response)" },
                                style = ForgeType.mono,
                                color = ForgeColors.inkMuted
                            )
                        }
                    }
                }
            }

            SectionCard("Actions") {
                ForgeButton(
                    if (active) "Open live training" else when (status) {
                        TrainingStatus.COMPLETED -> "Train again"
                        TrainingStatus.STOPPED, TrainingStatus.ERROR -> "Resume training"
                        else -> "Start training"
                    },
                    onClick = { vm.openTraining(modelId) },
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
                ForgeOutlineButton(
                    "Open playground and export",
                    onClick = { vm.openPlayground(modelId) },
                    modifier = Modifier.fillMaxWidth()
                )
                ForgeOutlineButton(
                    "Delete model",
                    onClick = { confirmDelete = true },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !active,
                    tint = ForgeColors.error
                )
                if (active) {
                    Text(
                        "Stop training before deleting this model.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
                    )
                }
                Text(
                    "The playground loads this model's best checkpoint or a packed export, " +
                        "generates text, and shares the packed file through Android's share sheet.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.verdigris
                )
            }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            containerColor = ForgeColors.surface,
            title = { Text("Delete ${manifest?.name ?: "model"}?", style = MaterialTheme.typography.headlineSmall) },
            text = {
                Text("This removes the model, datasets and all checkpoints from this device.")
            },
            confirmButton = {
                ForgeButton("Delete", onClick = {
                    confirmDelete = false
                    vm.deleteModel(modelId)
                })
            },
            dismissButton = {
                ForgeOutlineButton("Cancel", onClick = { confirmDelete = false })
            }
        )
    }
}
