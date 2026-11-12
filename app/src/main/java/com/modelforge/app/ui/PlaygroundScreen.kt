package com.modelforge.app.ui

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeBlue
import com.modelforge.app.ui.theme.ForgeDanger
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ForgeWarning
import java.io.File

@Composable
fun PlaygroundScreen(vm: ModelForgeViewModel, modelId: String) {
    val manifest = remember(modelId) { vm.manifest(modelId) }
    val state by vm.playground.collectAsState()
    val context = LocalContext.current

    ForgeScaffold(
        title = manifest?.name ?: "Playground",
        subtitle = "Generate text and export a packed model",
        onBack = { vm.openModel(modelId) }
    ) {
        if (manifest == null) {
            Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("This model's manifest is missing.", color = ForgeDanger)
                OutlinedButton(onClick = vm::goHome) { Text("Back to models") }
            }
            return@ForgeScaffold
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            SectionCard("Source") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = state.mode == PlaygroundState.MODE_CHECKPOINT,
                        onClick = { vm.updatePlaygroundMode(PlaygroundState.MODE_CHECKPOINT) },
                        label = { Text("Best checkpoint") }
                    )
                    FilterChip(
                        selected = state.mode == PlaygroundState.MODE_PACKED,
                        onClick = { vm.updatePlaygroundMode(PlaygroundState.MODE_PACKED) },
                        label = { Text("Packed") }
                    )
                }

                if (state.mode == PlaygroundState.MODE_PACKED) {
                    Text(
                        "Packed bits",
                        style = MaterialTheme.typography.labelMedium,
                        color = ForgeMuted
                    )
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(2, 3, 4, 8).forEach { bits ->
                            FilterChip(
                                selected = state.bits == bits,
                                onClick = { vm.updatePlaygroundBits(bits) },
                                label = { Text("$bits-bit") }
                            )
                        }
                    }
                    val packedSize = vm.packedFileSize(modelId, state.bits)
                    InfoRow(
                        "Packed file",
                        if (packedSize > 0) formatBytes(packedSize) else "not exported yet",
                        if (packedSize > 0) ForgeAccent else ForgeMuted
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = vm::exportPacked,
                            enabled = !state.busy,
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Export ${state.bits}-bit")
                        }
                        if (packedSize > 0) {
                            OutlinedButton(
                                onClick = {
                                    sharePacked(context, vm.packedFile(modelId, state.bits))
                                },
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("Share file")
                            }
                        }
                    }
                    Text(
                        "Packed inference decodes weights per group on the fly instead of " +
                            "expanding the whole matrix to fp32. 4-bit is usually the best " +
                            "size/quality tradeoff.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                } else {
                    val bestMb = vm.checkpointMegabytes(modelId).second
                    InfoRow(
                        "Best checkpoint",
                        if (bestMb > 0) "$bestMb MB" else "missing",
                        if (bestMb > 0) ForgeAccent else ForgeWarning
                    )
                    Text(
                        "Uses the full-precision weights-only checkpoint saved during training. " +
                            "No export step is needed, but the file is larger than a packed export.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                }
            }

            SectionCard("Prompt") {
                OutlinedTextField(
                    value = state.prompt,
                    onValueChange = vm::updatePlaygroundPrompt,
                    label = { Text("Prompt") },
                    minLines = 3,
                    maxLines = 6,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    "Max new tokens: ${state.maxNew}",
                    style = MaterialTheme.typography.labelMedium,
                    color = ForgeMuted
                )
                Slider(
                    value = state.maxNew.toFloat(),
                    onValueChange = { vm.updatePlaygroundMaxNew(it.toInt()) },
                    valueRange = 8f..256f,
                    steps = 30
                )
                Text(
                    "Temperature: ${String.format("%.2f", state.temperature)}",
                    style = MaterialTheme.typography.labelMedium,
                    color = ForgeMuted
                )
                Slider(
                    value = state.temperature,
                    onValueChange = vm::updatePlaygroundTemperature,
                    valueRange = 0.05f..1.5f,
                    steps = 28
                )
                Text(
                    "Top-k: ${state.topK}",
                    style = MaterialTheme.typography.labelMedium,
                    color = ForgeMuted
                )
                Slider(
                    value = state.topK.toFloat(),
                    onValueChange = { vm.updatePlaygroundTopK(it.toInt()) },
                    valueRange = 1f..100f,
                    steps = 24
                )
                Text(
                    "Top-p: ${String.format("%.2f", state.topP)}",
                    style = MaterialTheme.typography.labelMedium,
                    color = ForgeMuted
                )
                Slider(
                    value = state.topP,
                    onValueChange = vm::updatePlaygroundTopP,
                    valueRange = 0.05f..1f,
                    steps = 18
                )
                Button(
                    onClick = vm::generatePlayground,
                    enabled = !state.busy && state.prompt.isNotBlank(),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(if (state.mode == PlaygroundState.MODE_PACKED) "Generate from packed" else "Generate")
                }
                if (state.busy) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    if (state.status.isNotBlank()) {
                        Text(
                            state.status,
                            style = MaterialTheme.typography.bodySmall,
                            color = ForgeBlue
                        )
                    }
                }
            }

            if (state.error != null) {
                Text(state.error.orEmpty(), color = ForgeDanger, style = MaterialTheme.typography.bodyMedium)
            }

            if (state.output.isNotBlank()) {
                SectionCard("Output") {
                    SelectionContainer {
                        Text(
                            state.output,
                            style = MaterialTheme.typography.bodyMedium,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    InfoRow("Generated tokens", state.generatedTokens.toString())
                    InfoRow("Speed", formatTokensPerSecond(state.tokensPerSec), ForgeAccent)
                    Text(
                        "These models are trained from scratch on a phone. Repeating phrases " +
                            "and shaky grammar are expected; lower the temperature for safer text.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                }
            }
        }
    }
}

private fun sharePacked(context: Context, file: File) {
    if (!file.exists()) return
    runCatching {
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TITLE, file.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "Share ${file.name}"))
    }
}
