package com.modelforge.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val ForgeBackground = Color(0xFF0B0F14)
val ForgeSurface = Color(0xFF151C24)
val ForgeSurfaceHigh = Color(0xFF1E2833)
val ForgeAccent = Color(0xFF5BD6A2)
val ForgeAccentDim = Color(0xFF2E7D5B)
val ForgeBlue = Color(0xFF62A8FF)
val ForgeWarning = Color(0xFFFFC857)
val ForgeDanger = Color(0xFFFF6B6B)
val ForgeMuted = Color(0xFF8CA0B3)

private val ForgeColors = darkColorScheme(
    primary = ForgeAccent,
    onPrimary = Color(0xFF04120C),
    primaryContainer = ForgeAccentDim,
    onPrimaryContainer = Color(0xFFD9FFEF),
    secondary = ForgeBlue,
    onSecondary = Color(0xFF04121F),
    error = ForgeDanger,
    background = ForgeBackground,
    onBackground = Color(0xFFE6EEF6),
    surface = ForgeSurface,
    onSurface = Color(0xFFE6EEF6),
    surfaceVariant = ForgeSurfaceHigh,
    onSurfaceVariant = ForgeMuted,
    outline = Color(0xFF354555)
)

@Composable
fun ModelForgeTheme(
    @Suppress("UNUSED_PARAMETER") darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = ForgeColors,
        content = content
    )
}
