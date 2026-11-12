package com.modelforge.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.data.TrainingStatus
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.design.Hairline
import com.modelforge.app.ui.design.LedgerChart
import com.modelforge.app.ui.design.LedgerStat
import com.modelforge.app.ui.design.NotebookCard
import com.modelforge.app.ui.design.SmallCaps
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

@Composable
fun ForgeScaffold(
    title: String,
    subtitle: String? = null,
    onBack: (() -> Unit)? = null,
    actions: @Composable () -> Unit = {},
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ForgeColors.paper)
            .safeDrawingPadding()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    if (onBack != null) {
                        ForgeOutlineButton("← Back", onClick = onBack)
                    }
                    SmallCaps("MODELFORGE / WORKBENCH")
                }
                actions()
            }
            Text(
                title,
                style = MaterialTheme.typography.displaySmall,
                color = ForgeColors.ink,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
            if (!subtitle.isNullOrBlank()) {
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = ForgeColors.inkMuted,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        Hairline()
        Box(Modifier.weight(1f).fillMaxWidth()) {
            content()
        }
    }
}

@Composable
fun SectionCard(
    title: String? = null,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit
) {
    NotebookCard(modifier = modifier) {
        if (title != null) {
            SmallCaps(title)
            Hairline(color = ForgeColors.hairlineSoft)
        }
        content()
    }
}

@Composable
fun InfoRow(label: String, value: String, valueColor: Color = ForgeColors.ink) {
    LedgerStat(label, value, valueColor = valueColor)
}

@Composable
fun MetricTile(
    label: String,
    value: String,
    accent: Color = ForgeColors.ink,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(ForgeColors.surface, RoundedCornerShape(4.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        SmallCaps(label)
        Text(value, style = ForgeType.monoLarge, color = accent, maxLines = 1)
    }
}

@Composable
fun DropdownSelector(
    label: String,
    options: List<String>,
    selected: String,
    modifier: Modifier = Modifier,
    onSelect: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier = modifier) {
        ForgeOutlineButton(
            text = selected.ifBlank { label },
            onClick = { expanded = true },
            modifier = Modifier.widthIn(max = 280.dp)
        )
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option, style = MaterialTheme.typography.bodyMedium) },
                    onClick = {
                        onSelect(option)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
fun LossChart(history: List<Float>, modifier: Modifier = Modifier) {
    LedgerChart(history = history, modifier = modifier)
}

fun statusLabel(status: String): String = when (status) {
    TrainingStatus.PREPARING -> "Preparing"
    TrainingStatus.RUNNING -> "Training"
    TrainingStatus.PAUSED -> "Paused"
    TrainingStatus.STOPPED -> "Stopped"
    TrainingStatus.COMPLETED -> "Completed"
    TrainingStatus.ERROR -> "Error"
    else -> "Not started"
}

fun statusColor(status: String): Color = when (status) {
    TrainingStatus.COMPLETED -> ForgeColors.verdigris
    TrainingStatus.ERROR -> ForgeColors.error
    TrainingStatus.PAUSED -> ForgeColors.ochreInk
    TrainingStatus.RUNNING, TrainingStatus.PREPARING -> ForgeColors.emberText
    else -> ForgeColors.inkMuted
}

fun formatDuration(seconds: Long): String {
    if (seconds <= 0) return "—"
    val h = seconds / 3600
    val m = (seconds % 3600) / 60
    val s = seconds % 60
    return when {
        h > 0 -> "${h}h ${m}m"
        m > 0 -> "${m}m ${s}s"
        else -> "${s}s"
    }
}

fun formatLoss(value: Float): String =
    if (value <= 0f || value == Float.MAX_VALUE) "—" else String.format("%.4f", value)

fun formatTokensPerSecond(value: Double): String =
    if (value <= 0.0) "—" else String.format("%.0f tok/s", value)

fun formatParams(count: Long): String = ConfigDeriver.formatCount(count)

fun formatBytes(bytes: Long): String = ConfigDeriver.formatBytes(bytes)
