package com.modelforge.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.design.Hairline
import com.modelforge.app.ui.design.LedgerStat
import com.modelforge.app.ui.design.NotebookCard
import com.modelforge.app.ui.design.SectionHeading
import com.modelforge.app.ui.design.SmallCaps
import com.modelforge.app.ui.design.StatusTag
import com.modelforge.app.ui.design.TypographicEmptyState
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

@Composable
fun HomeScreen(vm: ModelForgeViewModel) {
    val models by vm.models.collectAsState()
    val tokenPresent by vm.tokenPresent.collectAsState()
    val notice by vm.notice.collectAsState()
    var showTokenDialog by remember { mutableStateOf(false) }

    ForgeScaffold(
        title = "The workbench",
        subtitle = "Train a small language model on this phone. No cloud, no hidden numbers.",
        actions = {
            ForgeOutlineButton(
                if (tokenPresent) "HF token · set" else "HF token",
                onClick = { showTokenDialog = true },
                tint = if (tokenPresent) ForgeColors.verdigris else ForgeColors.ink
            )
        }
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            item {
                NotebookCard {
                    SmallCaps("THIS DEVICE")
                    Text(
                        "A small model,\ntrained right here.",
                        style = MaterialTheme.typography.displayLarge,
                        color = ForgeColors.ink
                    )
                    Text(
                        vm.hardware.summary(),
                        style = MaterialTheme.typography.bodyMedium,
                        color = ForgeColors.inkMuted
                    )
                    Hairline()
                    Text(
                        formatParams(vm.hardware.maxTrainableParams),
                        style = ForgeType.displayNumber,
                        color = ForgeColors.emberText
                    )
                    SmallCaps("largest trainable model")
                    LedgerStat(
                        "battery",
                        if (vm.hardware.batteryPercent >= 0) "${vm.hardware.batteryPercent}%" else "unknown"
                    )
                    LedgerStat("thermal", vm.hardware.thermalStatus)
                    Text(
                        vm.hardware.limitReason,
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
                    )
                }
            }

            item {
                ForgeButton(
                    "New model",
                    onClick = vm::startNewModel,
                    modifier = Modifier.fillMaxWidth(),
                    primary = true
                )
            }

            if (notice != null) {
                item {
                    Text(
                        notice.orEmpty(),
                        color = ForgeColors.emberText,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }

            if (models.isEmpty()) {
                item {
                    NotebookCard {
                        TypographicEmptyState(
                            "No models on the bench.",
                            "Start with a name, a size, and one honest dataset."
                        )
                    }
                }
            } else {
                item {
                    SectionHeading("YOUR MODELS", "${models.size} built here")
                }
                items(models, key = { it.id }) { manifest ->
                    ModelListCard(manifest, vm)
                }
            }

            item {
                Hairline()
                Text(
                    "These are small models trained from scratch on a phone. Expect simple, " +
                        "short text, not chatbot-level answers. Every quality number is measured " +
                        "on the device.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
                ForgeOutlineButton("Design proof", onClick = vm::openGallery)
            }
        }
    }

    if (showTokenDialog) {
        TokenDialog(
            tokenPresent = tokenPresent,
            onDismiss = { showTokenDialog = false },
            onSave = { vm.saveToken(it); showTokenDialog = false },
            onClear = { vm.clearToken(); showTokenDialog = false }
        )
    }
}

@Composable
private fun ModelListCard(manifest: ModelManifest, vm: ModelForgeViewModel) {
    val state = vm.liveState(manifest.id)
    val status = state?.status ?: "not started"
    val active = state?.isActive == true
    NotebookCard(
        modifier = Modifier.clickable { vm.openModel(manifest.id) },
        selected = active
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                manifest.name,
                style = MaterialTheme.typography.headlineSmall,
                color = ForgeColors.ink
            )
            StatusTag(statusLabel(status), statusColor(status))
        }
        Text(
            "${manifest.config.layers} layers · ${manifest.config.bits}-bit · " +
                formatParams(manifest.config.paramCount()),
            style = ForgeType.monoSmall,
            color = ForgeColors.inkMuted
        )
        val total = state?.totalSteps ?: manifest.settings.steps
        val step = state?.step ?: 0
        if (step > 0 && total > 0) {
            StepProgress(step.toFloat() / total, statusColor(status))
            Text(
                "Step $step / $total" + (state?.loss?.takeIf { it > 0f }?.let {
                    " · loss ${formatLoss(it)}"
                } ?: ""),
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
        } else if (state?.error != null) {
            Text(
                state?.error.orEmpty(),
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.error,
                maxLines = 2
            )
        }
    }
}

@Composable
private fun StepProgress(fraction: Float, color: androidx.compose.ui.graphics.Color) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(3.dp)
                .background(ForgeColors.hairlineSoft, RoundedCornerShape(2.dp))
        ) {
            Box(
                Modifier
                    .fillMaxWidth(fraction.coerceIn(0f, 1f))
                    .height(3.dp)
                    .background(color, RoundedCornerShape(2.dp))
            )
        }
    }
}

@Composable
private fun TokenDialog(
    tokenPresent: Boolean,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit,
    onClear: () -> Unit
) {
    var token by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = ForgeColors.surface,
        title = { Text("Hugging Face token", style = MaterialTheme.typography.headlineSmall) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "Optional. Gated datasets need a token with read access. It is stored with " +
                        "EncryptedSharedPreferences and never written to logs.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
                OutlinedTextField(
                    value = token,
                    onValueChange = { token = it },
                    label = { Text("Access token") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation()
                )
                if (!tokenPresent) {
                    Text(
                        "No token saved.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
            }
        },
        confirmButton = {
            ForgeButton(
                "Save",
                onClick = { onSave(token) },
                enabled = token.isNotBlank()
            )
        },
        dismissButton = {
            Row {
                if (tokenPresent) {
                    ForgeOutlineButton("Clear", onClick = onClear, tint = ForgeColors.error)
                }
                ForgeOutlineButton("Cancel", onClick = onDismiss)
            }
        }
    )
}
