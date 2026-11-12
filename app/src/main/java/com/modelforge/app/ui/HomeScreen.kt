package com.modelforge.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeBlue
import com.modelforge.app.ui.theme.ForgeDanger
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ForgeSurface
import com.modelforge.app.ui.theme.ForgeSurfaceHigh
import com.modelforge.app.ui.theme.ForgeWarning

@Composable
fun HomeScreen(vm: ModelForgeViewModel) {
    val models by vm.models.collectAsState()
    val tokenPresent by vm.tokenPresent.collectAsState()
    val notice by vm.notice.collectAsState()
    var showTokenDialog by remember { mutableStateOf(false) }

    ForgeScaffold(
        title = "ModelForge",
        subtitle = "Train a small language model entirely on this phone.",
        actions = {
            TextButton(onClick = { showTokenDialog = true }) {
                Text(if (tokenPresent) "HF token ✓" else "HF token", color = ForgeBlue)
            }
        }
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                SectionCard("This device") {
                    Text(vm.hardware.summary(), style = MaterialTheme.typography.bodyMedium)
                    InfoRow(
                        "Battery",
                        if (vm.hardware.batteryPercent >= 0) "${vm.hardware.batteryPercent}%" else "unknown"
                    )
                    InfoRow("Thermal", vm.hardware.thermalStatus)
                    InfoRow(
                        "Largest trainable model",
                        formatParams(vm.hardware.maxTrainableParams),
                        ForgeAccent
                    )
                    Text(
                        vm.hardware.limitReason,
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted
                    )
                }
            }

            item {
                Button(onClick = vm::startNewModel, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Filled.Add, contentDescription = null)
                    Text("  New model")
                }
            }

            if (notice != null) {
                item {
                    Text(
                        notice.orEmpty(),
                        color = ForgeAccent,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }

            if (models.isEmpty()) {
                item {
                    SectionCard("No models yet") {
                        Text(
                            "Create a model, pick a dataset from Hugging Face, and start training. " +
                                "Checkpoints let you pause and resume without losing progress.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = ForgeMuted
                        )
                    }
                }
            } else {
                item {
                    Text(
                        "Your models",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
                items(models, key = { it.id }) { manifest ->
                    ModelListCard(manifest, vm)
                }
            }

            item {
                Text(
                    "These are small models trained from scratch on a phone. Expect simple, " +
                        "short text, not chatbot-level answers. Every quality number shown in the app " +
                        "is measured on the device.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeMuted,
                    modifier = Modifier.padding(top = 8.dp, bottom = 24.dp)
                )
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
    val colors = if (state?.isActive == true) ForgeSurfaceHigh else ForgeSurface
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { vm.openModel(manifest.id) },
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = colors)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    manifest.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(statusLabel(status), color = statusColor(status), style = MaterialTheme.typography.labelMedium)
            }
            Text(
                "${formatParams(manifest.config.paramCount())} parameters • " +
                    "${manifest.config.layers} layers • ${manifest.config.bits}-bit",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeMuted
            )
            val total = state?.totalSteps ?: manifest.settings.steps
            val step = state?.step ?: 0
            if (step > 0 && total > 0) {
                LinearProgressIndicator(
                    progress = (step.toFloat() / total).coerceIn(0f, 1f),
                    modifier = Modifier.fillMaxWidth(),
                    color = statusColor(status),
                    trackColor = ForgeSurfaceHigh
                )
                Text(
                    "Step $step / $total" + (state?.loss?.takeIf { it > 0f }?.let {
                        " • loss ${formatLoss(it)}"
                    } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeMuted
                )
            } else if (state?.error != null) {
                Text(
                    state?.error.orEmpty(),
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeDanger,
                    maxLines = 2
                )
            }
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
        title = { Text("Hugging Face token") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "Optional. Gated datasets need a token with read access. It is stored with " +
                        "EncryptedSharedPreferences and never written to logs.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeMuted
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
                        color = ForgeWarning
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onSave(token) },
                enabled = token.isNotBlank()
            ) { Text("Save") }
        },
        dismissButton = {
            Row {
                if (tokenPresent) {
                    TextButton(onClick = onClear) { Text("Clear", color = ForgeDanger) }
                }
                TextButton(onClick = onDismiss) { Text("Cancel") }
            }
        }
    )
}
