package com.modelforge.app.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * The single source of truth for the ModelForge v2 notebook palette.
 * Screens must use these tokens (directly or through MaterialTheme), never raw hex.
 */
object ForgeColors {
    val paper = Color(0xFFF4EFE6)
    val surface = Color(0xFFFBF8F2)
    val ink = Color(0xFF1B1A17)
    val inkMuted = Color(0xFF6B655A)
    val hairline = Color(0xFFDDD5C6)
    val hairlineSoft = Color(0xFFE9E2D5)
    val ember = Color(0xFFD9481C)
    val emberText = Color(0xFFB83A12)
    val verdigris = Color(0xFF1F5C54)
    val ochre = Color(0xFFB8860B)
    val ochreInk = Color(0xFF7D5906)
    val ochreTint = Color(0xFFF3E3B3)
    val error = Color(0xFFA8281E)
    val errorTint = Color(0xFFF6D9D4)
    val inkGhost = Color(0xFF6F685C)

    /** Light ink wash used for selected rows and tonal layers. */
    val inkWash = Color(0xFFECE5D8)
    val paperDeep = Color(0xFFEDE5D8)
}

private val NotebookColors = lightColorScheme(
    primary = ForgeColors.emberText,
    onPrimary = ForgeColors.paper,
    primaryContainer = ForgeColors.ochreTint,
    onPrimaryContainer = ForgeColors.ink,
    secondary = ForgeColors.verdigris,
    onSecondary = ForgeColors.paper,
    secondaryContainer = ForgeColors.inkWash,
    onSecondaryContainer = ForgeColors.ink,
    tertiary = ForgeColors.ochreInk,
    onTertiary = ForgeColors.paper,
    background = ForgeColors.paper,
    onBackground = ForgeColors.ink,
    surface = ForgeColors.surface,
    onSurface = ForgeColors.ink,
    surfaceVariant = ForgeColors.inkWash,
    onSurfaceVariant = ForgeColors.inkMuted,
    outline = ForgeColors.hairline,
    outlineVariant = ForgeColors.hairlineSoft,
    error = ForgeColors.error,
    onError = ForgeColors.paper,
    errorContainer = ForgeColors.errorTint,
    onErrorContainer = ForgeColors.error
)

private val NotebookShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(4.dp),
    medium = RoundedCornerShape(6.dp),
    large = RoundedCornerShape(6.dp),
    extraLarge = RoundedCornerShape(6.dp)
)

@Composable
fun ModelForgeTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = NotebookColors,
        typography = ModelForgeTypography,
        shapes = NotebookShapes,
        content = content
    )
}

