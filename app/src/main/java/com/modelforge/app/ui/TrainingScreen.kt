package com.modelforge.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingStatus
import kotlinx.coroutines.delay
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

@Composable
fun TrainingScreen(vm: ModelForgeViewModel, modelId: String) {
    val manifest = remember(modelId) { vm.manifest(modelId) }
    var state by remember(modelId) { mutableStateOf(vm.liveState(modelId)) }

    LaunchedEffect(modelId) {
        while (true) {
            state = vm.liveState(modelId)
            delay(1000)
        }
    }

    val status = state?.status ?: "not started"
    ForgeScaffold(
        title = manifest?.name ?: "Training",
        subtitle = statusLabel(status),
        onBack = vm::goHome
    ) {
        if (manifest == null) {
            Column(Modifier.fillMaxSize().padding(24.dp)) {
                Text("This model's manifest is missing.", color = ForgeColors.error)
                ForgeOutlineButton("Back to models", onClick = vm::goHome)
            }
            return@ForgeScaffold
        }

        val snapshot = state
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            ProgressCard(manifest, snapshot)
            if (snapshot != null) {
                MetricsRow(snapshot)
            }
            SectionCard("Loss curve") {
                LossChart(
                    history = snapshot?.lossHistory.orEmpty(),
                    modifier = Modifier.fillMaxWidth().height(160.dp)
                )
            }
            if (snapshot != null) {
                DeviceCard(snapshot)
                if (snapshot.sample.isNotBlank()) {
                    SectionCard("Sample output") {
                        Text(
                            snapshot.sample,
                            style = ForgeType.mono,
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(ForgeColors.inkWash, RoundedCornerShape(4.dp))
                                .padding(12.dp)
                        )
                        Text(
                            "Greedy-ish sample from the live model. Tiny models repeat and " +
                                "ramble; that is expected.",
                            style = MaterialTheme.typography.bodySmall,
                            color = ForgeColors.inkMuted
                        )
                    }
                }
            }
            if (snapshot?.error != null) {
                Text(snapshot?.error.orEmpty(), color = ForgeColors.error, style = MaterialTheme.typography.bodyMedium)
            }
            ControlsCard(vm, modelId, manifest, snapshot)
        }
    }
}

@Composable
private fun ProgressCard(manifest: ModelManifest, state: TrainingLiveState?) {
    val total = state?.totalSteps?.takeIf { it > 0 } ?: manifest.settings.steps
    val step = state?.step ?: 0
    SectionCard("Progress") {
        Text(
            "${statusLabel(state?.status ?: "")} • step $step of $total",
            style = MaterialTheme.typography.titleMedium
        )
        Box(
            Modifier
                .fillMaxWidth()
                .height(3.dp)
                .background(ForgeColors.hairlineSoft, RoundedCornerShape(2.dp))
        ) {
            Box(
                Modifier
                    .fillMaxWidth((step.toFloat() / total).coerceIn(0f, 1f))
                    .height(3.dp)
                    .background(statusColor(state?.status ?: ""), RoundedCornerShape(2.dp))
            )
        }
        InfoRow("Elapsed", formatDuration((state?.elapsedMs ?: 0L) / 1000L))
        InfoRow("ETA", formatDuration(state?.etaSeconds ?: 0L))
        if (state?.autoPaused == true) {
            Text(
                "Training is auto-paused to protect the phone. It resumes when battery and " +
                    "temperature are safe.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.ochreInk
            )
        }
        val message = state?.message.orEmpty()
        if (message.isNotBlank()) {
            Text(
                message,
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
    }
}

@Composable
private fun MetricsRow(state: TrainingLiveState) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        MetricTile("Loss", formatLoss(state.loss), ForgeColors.emberText, Modifier.weight(1f))
        MetricTile("Best loss", formatLoss(state.bestLoss), ForgeColors.verdigris, Modifier.weight(1f))
        MetricTile("Speed", formatTokensPerSecond(state.tokensPerSec), ForgeColors.emberText, Modifier.weight(1f))
        MetricTile("RAM", if (state.ramMb > 0) "${state.ramMb} MB" else "—", ForgeColors.inkMuted, Modifier.weight(1f))
    }
}

@Composable
private fun DeviceCard(state: TrainingLiveState) {
    SectionCard("Device") {
        InfoRow("Battery", if (state.batteryPercent >= 0) "${state.batteryPercent}%" else "unknown")
        InfoRow("Thermal", state.thermalStatus.ifBlank { "unknown" })
        InfoRow("Batch size", state.batchSize.toString())
        state.deviceTempC?.let { InfoRow("Battery temperature", String.format("%.1f °C", it)) }
    }
}

@Composable
private fun ControlsCard(
    vm: ModelForgeViewModel,
    modelId: String,
    manifest: ModelManifest,
    state: TrainingLiveState?
) {
    val status = state?.status
    val active = state?.isActive == true
    SectionCard("Controls") {
        when {
            active && status == TrainingStatus.PAUSED -> {
                ForgeButton("Resume training", onClick = vm::resumeTraining, modifier = Modifier.fillMaxWidth(), primary = true)
            }
            active -> {
                ForgeOutlineButton("Pause training", onClick = vm::pauseTraining, modifier = Modifier.fillMaxWidth())
            }
            else -> {
                ForgeButton(
                    if (status == TrainingStatus.COMPLETED) "Train again" else "Start training",
                    onClick = { vm.openTraining(modelId) },
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
                ForgeOutlineButton(
                    "Open model details",
                    onClick = { vm.openModel(modelId) },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
        if (active) {
            ForgeOutlineButton("Stop training", onClick = vm::stopTraining, modifier = Modifier.fillMaxWidth(), tint = ForgeColors.error)
            Text(
                "Training continues if you leave the app. Use the notification to pause or stop.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
        }
        if (status == TrainingStatus.COMPLETED) {
            Text(
                "Training completed. The best checkpoint is kept for the playground.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.emberText
            )
        }
        InfoRow("Best checkpoint", if (vm.checkpointMegabytes(modelId).second > 0) "saved" else "not yet")
        InfoRow("Steps configured", manifest.settings.steps.toString())
    }
}
