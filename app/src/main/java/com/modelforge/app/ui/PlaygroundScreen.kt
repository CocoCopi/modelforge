package com.modelforge.app.ui

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import java.io.File
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

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
                Text("This model's manifest is missing.", color = ForgeColors.error)
                ForgeOutlineButton("Back to models", onClick = vm::goHome)
            }
            return@ForgeScaffold
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            SectionCard("Source") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Choice(
                        text = "Best checkpoint",
                        selected = state.mode == PlaygroundState.MODE_CHECKPOINT,
                        onClick = { vm.updatePlaygroundMode(PlaygroundState.MODE_CHECKPOINT) }
                    )
                    Choice(
                        text = "Packed",
                        selected = state.mode == PlaygroundState.MODE_PACKED,
                        onClick = { vm.updatePlaygroundMode(PlaygroundState.MODE_PACKED) }
                    )
                }

                if (state.mode == PlaygroundState.MODE_PACKED) {
                    Text(
                        "Packed bits",
                        style = MaterialTheme.typography.labelMedium,
                        color = ForgeColors.inkMuted
                    )
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        listOf(2, 3, 4, 8).forEach { bits ->
                            Choice(
                                text = "$bits-bit",
                                selected = state.bits == bits,
                                onClick = { vm.updatePlaygroundBits(bits) }
                            )
                        }
                    }
                    val packedSize = vm.packedFileSize(modelId, state.bits)
                    InfoRow(
                        "Packed file",
                        if (packedSize > 0) formatBytes(packedSize) else "not exported yet",
                        if (packedSize > 0) ForgeColors.emberText else ForgeColors.inkMuted
                    )
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        ForgeButton(
                            "Export ${state.bits}-bit",
                            onClick = vm::exportPacked,
                            enabled = !state.busy,
                            modifier = Modifier.weight(1f),
                            primary = true
                        )
                        if (packedSize > 0) {
                            ForgeOutlineButton(
                                "Share file",
                                onClick = {
                                    sharePacked(context, vm.packedFile(modelId, state.bits))
                                },
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                    Text(
                        "Packed inference decodes weights per group on the fly instead of " +
                            "expanding the whole matrix to fp32. 4-bit is usually the best " +
                            "size/quality tradeoff.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
                    )
                } else {
                    val bestMb = vm.checkpointMegabytes(modelId).second
                    InfoRow(
                        "Best checkpoint",
                        if (bestMb > 0) "$bestMb MB" else "missing",
                        if (bestMb > 0) ForgeColors.emberText else ForgeColors.ochreInk
                    )
                    Text(
                        "Uses the full-precision weights-only checkpoint saved during training. " +
                            "No export step is needed, but the file is larger than a packed export.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
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
                    color = ForgeColors.inkMuted
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
                    color = ForgeColors.inkMuted
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
                    color = ForgeColors.inkMuted
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
                    color = ForgeColors.inkMuted
                )
                Slider(
                    value = state.topP,
                    onValueChange = vm::updatePlaygroundTopP,
                    valueRange = 0.05f..1f,
                    steps = 18
                )
                ForgeButton(
                    if (state.mode == PlaygroundState.MODE_PACKED) "Generate from packed" else "Generate",
                    onClick = vm::generatePlayground,
                    enabled = !state.busy && state.prompt.isNotBlank(),
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
                if (state.busy) {
                    Text("generating…", style = ForgeType.mono, color = ForgeColors.emberText)
                    if (state.status.isNotBlank()) {
                        Text(
                            state.status,
                            style = MaterialTheme.typography.bodySmall,
                            color = ForgeColors.verdigris
                        )
                    }
                }
            }

            if (state.error != null) {
                Text(state.error.orEmpty(), color = ForgeColors.error, style = MaterialTheme.typography.bodyMedium)
            }

            if (state.output.isNotBlank()) {
                SectionCard("Output") {
                    SelectionContainer {
                        Text(
                            state.output,
                            style = ForgeType.mono
                        )
                    }
                    InfoRow("Generated tokens", state.generatedTokens.toString())
                    InfoRow("Speed", formatTokensPerSecond(state.tokensPerSec), ForgeColors.emberText)
                    Text(
                        "These models are trained from scratch on a phone. Repeating phrases " +
                            "and shaky grammar are expected; lower the temperature for safer text.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
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

@Composable
private fun Choice(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    androidx.compose.foundation.layout.Box(
        modifier = modifier
            .heightIn(min = 40.dp)
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
        contentAlignment = androidx.compose.ui.Alignment.Center
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge, color = if (selected) ForgeColors.emberText else ForgeColors.ink)
    }
}
