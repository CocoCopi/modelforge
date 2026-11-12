package com.modelforge.app.ui.design

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToLong

val NotebookShape = RoundedCornerShape(6.dp)

@Composable
fun NotebookCard(
    modifier: Modifier = Modifier,
    selected: Boolean = false,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(
                if (selected) ForgeColors.inkWash else ForgeColors.surface,
                NotebookShape
            )
            .border(
                1.dp,
                if (selected) ForgeColors.emberText else ForgeColors.hairline,
                NotebookShape
            )
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        content = content
    )
}

@Composable
fun SmallCaps(text: String, color: Color = ForgeColors.inkMuted, modifier: Modifier = Modifier) {
    Text(
        text.uppercase(),
        modifier = modifier,
        style = MaterialTheme.typography.labelSmall,
        color = color
    )
}

@Composable
fun Hairline(modifier: Modifier = Modifier, color: Color = ForgeColors.hairline) {
    Box(
        modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(color)
    )
}

@Composable
fun SectionHeading(text: String, trailing: String? = null) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Bottom
    ) {
        SmallCaps(text)
        if (trailing != null) {
            Text(trailing, style = ForgeType.monoSmall, color = ForgeColors.inkMuted)
        }
    }
    Hairline()
}

@Composable
fun ForgeButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    primary: Boolean = false,
    enabled: Boolean = true
) {
    val fill = when {
        !enabled -> ForgeColors.inkWash
        primary -> ForgeColors.emberText
        else -> ForgeColors.ink
    }
    val label = when {
        !enabled -> ForgeColors.inkGhost
        else -> ForgeColors.paper
    }
    Box(
        modifier = modifier
            .heightIn(min = 48.dp)
            .background(fill, RoundedCornerShape(4.dp))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 14.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = label,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun ForgeOutlineButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    tint: Color = ForgeColors.ink
) {
    Box(
        modifier = modifier
            .heightIn(min = 48.dp)
            .border(1.dp, if (enabled) ForgeColors.hairline else ForgeColors.hairlineSoft, RoundedCornerShape(4.dp))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = 18.dp, vertical = 14.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = if (enabled) tint else ForgeColors.inkGhost,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun LedgerStat(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    valueColor: Color = ForgeColors.ink
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = ForgeColors.inkMuted)
        Text(value, style = ForgeType.mono, color = valueColor, textAlign = TextAlign.End)
    }
}

@Composable
fun StatusTag(text: String, tint: Color = ForgeColors.verdigris) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelSmall,
        color = tint,
        modifier = Modifier
            .border(1.dp, tint.copy(alpha = 0.55f), RoundedCornerShape(3.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    )
}

@Composable
fun TypographicEmptyState(phrase: String, guidance: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(phrase, style = MaterialTheme.typography.displaySmall, color = ForgeColors.ink)
        Text(guidance, style = MaterialTheme.typography.bodyMedium, color = ForgeColors.inkMuted)
    }
}

/** Log-scale parameter ruler with decade ticks, a big rolling number and haptics. */
@Composable
fun ParameterRuler(
    value: Long,
    min: Long,
    max: Long,
    onValueChange: (Long) -> Unit,
    modifier: Modifier = Modifier
) {
    val haptic = LocalHapticFeedback.current
    var lastDecade by remember { mutableIntStateOf(decadeOf(value)) }
    val animated by animateFloatAsState(
        targetValue = value.toFloat(),
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioNoBouncy,
            stiffness = Spring.StiffnessMediumLow
        ),
        label = "parameter-roll"
    )

    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            ConfigDeriver.formatCount(animated.roundToLong().coerceAtLeast(1L)),
            style = ForgeType.displayNumber,
            color = ForgeColors.ink
        )
        SmallCaps("parameters")
        Spacer(Modifier.height(2.dp))
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
                .pointerInput(min, max) {
                    val width = size.width.toFloat().coerceAtLeast(1f)
                    fun commit(x: Float) {
                        val fraction = (x / width).coerceIn(0f, 1f)
                        val raw = valueAt(fraction, min, max)
                        val snapped = snap(raw, min, max)
                        val decade = decadeOf(snapped)
                        if (decade != lastDecade) {
                            lastDecade = decade
                            haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                        }
                        onValueChange(snapped)
                    }
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        commit(down.position.x)
                        down.consume()
                        while (true) {
                            val event = awaitPointerEvent()
                            val change = event.changes.firstOrNull { it.id == down.id } ?: break
                            if (!change.pressed) break
                            commit(change.position.x)
                            change.consume()
                        }
                    }
                }
        ) {
            val baseline = size.height * 0.72f
            val width = size.width
            drawLine(
                ForgeColors.hairline,
                Offset(0f, baseline),
                Offset(width, baseline),
                strokeWidth = 1.5f
            )
            val minors = 80
            for (i in 0..minors) {
                val f = i / minors.toFloat()
                val x = width * f
                val decadeTick = abs(f - 0.5f) < 1e-4f || i % 20 == 0
                val h = if (decadeTick) 16f else 7f
                drawLine(
                    if (decadeTick) ForgeColors.inkMuted else ForgeColors.hairline,
                    Offset(x, baseline - h),
                    Offset(x, baseline),
                    strokeWidth = if (decadeTick) 1.5f else 1f
                )
            }
            val fraction = fractionOf(value, min, max)
            val x = width * fraction
            drawLine(
                ForgeColors.ember,
                Offset(x, 4f),
                Offset(x, baseline + 10f),
                strokeWidth = 2.5f
            )
            val mark = Path().apply {
                moveTo(x - 7f, baseline + 10f)
                lineTo(x + 7f, baseline + 10f)
                lineTo(x, baseline + 18f)
                close()
            }
            drawPath(mark, ForgeColors.ember)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(ConfigDeriver.formatCount(min), style = ForgeType.monoSmall, color = ForgeColors.inkMuted)
            Text(ConfigDeriver.formatCount(max), style = ForgeType.monoSmall, color = ForgeColors.inkMuted)
        }
    }
}

@Composable
fun BitGrainSelector(
    bits: Int,
    sizes: Map<Int, Long>,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val haptic = LocalHapticFeedback.current
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        listOf(32, 16, 8, 4, 3, 2).forEach { option ->
            val selected = option == bits
            Column(
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 72.dp)
                    .background(
                        if (selected) ForgeColors.inkWash else ForgeColors.surface,
                        RoundedCornerShape(4.dp)
                    )
                    .border(
                        1.dp,
                        if (selected) ForgeColors.emberText else ForgeColors.hairline,
                        RoundedCornerShape(4.dp)
                    )
                    .clickable {
                        haptic.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        onSelect(option)
                    }
                    .padding(vertical = 6.dp, horizontal = 4.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(3.dp)
            ) {
                Canvas(Modifier.fillMaxWidth().height(20.dp)) {
                    val density = when (option) {
                        32 -> 0.94f
                        16 -> 0.78f
                        8 -> 0.58f
                        4 -> 0.4f
                        3 -> 0.3f
                        else -> 0.2f
                    }
                    val cols = 14
                    val rows = 5
                    for (row in 0 until rows) {
                        for (col in 0 until cols) {
                            val seed = (row * 31 + col * 17 + option * 13) % 100 / 100f
                            if (seed <= density) {
                                val x = size.width * (col + 0.5f) / cols
                                val y = size.height * (row + 0.5f) / rows
                                drawCircle(
                                    if (selected) ForgeColors.emberText else ForgeColors.inkMuted,
                                    radius = 1.2f,
                                    center = Offset(x, y)
                                )
                            }
                        }
                    }
                }
                Text(
                    "$option",
                    style = ForgeType.mono,
                    fontWeight = FontWeight.Medium,
                    color = if (selected) ForgeColors.emberText else ForgeColors.ink
                )
                Text(
                    sizes[option]?.let { ConfigDeriver.formatBytes(it) } ?: "—",
                    style = ForgeType.monoSmall,
                    color = ForgeColors.inkMuted
                )
            }
        }
    }
}

@Composable
fun WizardRuler(step: Int, total: Int, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            SmallCaps("STEP ${(step + 1).toString().padStart(2, '0')} / $total")
            if (total > 0) {
                Text(
                    "${((step + 1) * 100 / total)}%",
                    style = ForgeType.monoSmall,
                    color = ForgeColors.inkMuted
                )
            }
        }
        Canvas(Modifier.fillMaxWidth().height(16.dp)) {
            val baseline = size.height - 4f
            drawLine(ForgeColors.hairline, Offset(0f, baseline), Offset(size.width, baseline), 1f)
            for (i in 0 until total) {
                val x = if (total == 1) 0f else size.width * i / (total - 1).toFloat()
                val active = i == step
                val done = i < step
                drawLine(
                    when {
                        active -> ForgeColors.ember
                        done -> ForgeColors.inkMuted
                        else -> ForgeColors.hairline
                    },
                    Offset(x, baseline - if (active) 12f else 6f),
                    Offset(x, baseline),
                    strokeWidth = if (active) 2.5f else 1f
                )
            }
        }
    }
}

@Composable
fun LedgerChart(
    history: List<Float>,
    modifier: Modifier = Modifier,
    annotations: List<Pair<Int, String>> = emptyList()
) {
    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        if (history.size < 2) {
            Text(
                "The ledger waits for a few steps.",
                style = MaterialTheme.typography.bodySmall,
                color = ForgeColors.inkMuted
            )
            return@Column
        }
        val min = history.min()
        val max = history.max()
        val range = (max - min).coerceAtLeast(1e-6f)
        Canvas(Modifier.fillMaxWidth().height(150.dp)) {
            repeat(4) { i ->
                val y = size.height * i / 4f
                drawLine(
                    ForgeColors.hairlineSoft,
                    Offset(0f, y),
                    Offset(size.width, y),
                    strokeWidth = 1f
                )
            }
            val path = Path()
            history.forEachIndexed { index, value ->
                val x = size.width * index / (history.size - 1).toFloat()
                val y = size.height * (1f - (value - min) / range)
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, ForgeColors.ink, style = Stroke(width = 2f, cap = StrokeCap.Round))
            val lastX = size.width
            val lastY = size.height * (1f - (history.last() - min) / range)
            drawCircle(ForgeColors.ember, radius = 4f, center = Offset(lastX, lastY))
            annotations.forEach { (index, _) ->
                if (index in history.indices) {
                    val x = size.width * index / (history.size - 1).toFloat()
                    drawLine(ForgeColors.ochre, Offset(x, 0f), Offset(x, size.height), 1f)
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                "min ${String.format("%.4f", min)}",
                style = ForgeType.monoSmall,
                color = ForgeColors.inkMuted
            )
            if (annotations.isNotEmpty()) {
                Text(annotations.last().second, style = ForgeType.monoSmall, color = ForgeColors.ochreInk)
            }
            Text(
                "now ${String.format("%.4f", history.last())}",
                style = ForgeType.monoSmall,
                color = ForgeColors.emberText
            )
        }
    }
}

@Composable
fun InkAvatar(initials: String, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .size(40.dp)
            .border(1.dp, ForgeColors.ink, RoundedCornerShape(4.dp)),
        contentAlignment = Alignment.Center
    ) {
        Text(initials.uppercase().take(2), style = ForgeType.mono, color = ForgeColors.ink)
    }
}

private fun decadeOf(value: Long): Int = when {
    value <= 0 -> 0
    value < 1_000 -> 3
    else -> value.toString().length
}

private fun fractionOf(value: Long, min: Long, max: Long): Float {
    if (max <= min) return 0f
    val v = value.coerceIn(min, max).toDouble()
    return ((ln(v) - ln(min.toDouble())) / (ln(max.toDouble()) - ln(min.toDouble()))).toFloat()
}

private fun valueAt(fraction: Float, min: Long, max: Long): Long {
    val v = exp(ln(min.toDouble()) + fraction * (ln(max.toDouble()) - ln(min.toDouble())))
    return v.roundToLong().coerceIn(min, max)
}

private fun snap(raw: Long, min: Long, max: Long): Long {
    val candidates = mutableListOf(min, max)
    var exponent = 0
    while (true) {
        val base = 10.0.pow(exponent)
        if (base > max) break
        for (m in listOf(1.0, 2.0, 3.0, 5.0, 7.5)) {
            val candidate = (m * base).roundToLong()
            if (candidate in min..max) candidates.add(candidate)
        }
        exponent++
    }
    return candidates.minByOrNull { abs(it - raw) } ?: raw
}
