package com.modelforge.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.theme.ForgeColors

@Composable
fun AiKeyDialog(
    keyPresent: Boolean,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit,
    onClear: () -> Unit
) {
    var key by remember { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = ForgeColors.surface,
        title = { Text("NVIDIA API key", style = MaterialTheme.typography.headlineSmall) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    "Used only for the AI architect. It is stored with EncryptedSharedPreferences, " +
                        "sent only to integrate.api.nvidia.com, and never written to logs or the model card.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ForgeColors.inkMuted
                )
                OutlinedTextField(
                    value = key,
                    onValueChange = { key = it },
                    label = { Text("API key") },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation()
                )
                if (!keyPresent) {
                    Text(
                        "No NVIDIA key saved.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.ochreInk
                    )
                }
            }
        },
        confirmButton = {
            ForgeButton(
                "Save",
                onClick = { onSave(key) },
                enabled = key.isNotBlank()
            )
        },
        dismissButton = {
            Row {
                if (keyPresent) {
                    ForgeOutlineButton("Clear", onClick = onClear, tint = ForgeColors.error)
                }
                ForgeOutlineButton("Cancel", onClick = onDismiss)
            }
        }
    )
}
