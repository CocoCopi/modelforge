package com.modelforge.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.data.TrainingStatus
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeBlue
import com.modelforge.app.ui.theme.ForgeDanger
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ForgeSurface
import com.modelforge.app.ui.theme.ForgeSurfaceHigh
import com.modelforge.app.ui.theme.ForgeWarning
import kotlin.math.max

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
            .background(MaterialTheme.colorScheme.background)
            .safeDrawingPadding()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (onBack != null) {
                IconButton(onClick = onBack) {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = MaterialTheme.colorScheme.onBackground
                    )
                }
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    title,
                    style = MaterialTheme.typography.headlineSmall,
                    color = ForgeAccent,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                if (!subtitle.isNullOrBlank()) {
                    Text(
                        subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeMuted,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
            actions()
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(MaterialTheme.colorScheme.outline))
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
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = ForgeSurface)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (title != null) {
                Text(title, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurface)
            }
            content()
        }
    }
}

@Composable
fun InfoRow(label: String, value: String, valueColor: Color = MaterialTheme.colorScheme.onSurface) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = ForgeMuted)
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            color = valueColor,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
fun MetricTile(
    label: String,
    value: String,
    accent: Color = ForgeAccent,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(ForgeSurfaceHigh, RoundedCornerShape(12.dp))
            .padding(horizontal = 10.dp, vertical = 8.dp)
    ) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = ForgeMuted)
        Text(value, style = MaterialTheme.typography.titleMedium, color = accent, maxLines = 1)
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
        OutlinedButton(onClick = { expanded = true }) {
            Text(
                selected.ifBlank { label },
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Icon(Icons.Filled.ArrowDropDown, contentDescription = null)
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option) },
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
    Box(
        modifier = modifier
            .background(ForgeSurfaceHigh, RoundedCornerShape(12.dp))
            .padding(10.dp)
    ) {
        if (history.size < 2) {
            Text(
                "The loss curve appears after a few steps.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeMuted
            )
        } else {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val min = history.min()
                val max = history.max()
                val range = max(max - min, 1e-6f)
                var previous: Offset? = null
                history.forEachIndexed { index, value ->
                    val x = size.width * index / (history.size - 1).toFloat()
                    val y = size.height * (1f - (value - min) / range)
                    val point = Offset(x, y)
                    previous?.let {
                        drawLine(
                            color = ForgeAccent,
                            start = it,
                            end = point,
                            strokeWidth = 3f
                        )
                    }
                    previous = point
                }
            }
        }
    }
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
    TrainingStatus.COMPLETED -> ForgeAccent
    TrainingStatus.ERROR -> ForgeDanger
    TrainingStatus.PAUSED -> ForgeWarning
    TrainingStatus.RUNNING, TrainingStatus.PREPARING -> ForgeBlue
    else -> ForgeMuted
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
