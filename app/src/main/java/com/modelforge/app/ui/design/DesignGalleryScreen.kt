package com.modelforge.app.ui.design

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType

@Composable
fun DesignGalleryScreen(onBack: () -> Unit = {}) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ForgeColors.paper)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom
        ) {
            Column {
                SmallCaps("MODELFORGE / DESIGN PROOF")
                Text("The notebook", style = MaterialTheme.typography.displayMedium, color = ForgeColors.ink)
            }
            ForgeOutlineButton("Back", onBack)
        }
        Hairline()

        SectionHeading("Color tokens", "paper, ink, one ember")
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Swatch("paper", ForgeColors.paper, ForgeColors.ink)
            Swatch("surface", ForgeColors.surface, ForgeColors.ink)
            Swatch("ink", ForgeColors.ink, ForgeColors.paper)
            Swatch("ember", ForgeColors.ember, ForgeColors.paper)
            Swatch("verdigris", ForgeColors.verdigris, ForgeColors.paper)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Swatch("ochre", ForgeColors.ochre, ForgeColors.ink)
            Swatch("ochreInk", ForgeColors.ochreInk, ForgeColors.paper)
            Swatch("error", ForgeColors.error, ForgeColors.paper)
            Swatch("hairline", ForgeColors.hairline, ForgeColors.ink)
            Swatch("inkMuted", ForgeColors.inkMuted, ForgeColors.paper)
        }

        SectionHeading("Type scale", "Fraunces / Instrument Sans / JetBrains Mono")
        NotebookCard {
            Text("Editorial title", style = MaterialTheme.typography.displayMedium, color = ForgeColors.ink)
            Text(
                "Body text is set in Instrument Sans with generous leading for calm reading.",
                style = MaterialTheme.typography.bodyLarge,
                color = ForgeColors.ink
            )
            Text("momentum 0.0042   step 1280/2000   tokens/s 81,442", style = ForgeType.mono, color = ForgeColors.inkMuted)
            SmallCaps("a small caps label")
        }

        SectionHeading("Parameter ruler", "log scale, decade haptics")
        NotebookCard {
            ParameterRuler(
                value = 5_000_000L,
                min = 1_000_000L,
                max = 50_000_000L,
                onValueChange = {}
            )
        }

        SectionHeading("Bit grain", "density reflects precision")
        NotebookCard {
            BitGrainSelector(
                bits = 4,
                sizes = mapOf(
                    32 to 80_000_000L,
                    16 to 40_000_000L,
                    8 to 20_000_000L,
                    4 to 10_000_000L,
                    3 to 7_500_000L,
                    2 to 5_000_000L
                ),
                onSelect = {}
            )
        }

        SectionHeading("Wizard ruler", "one notch per decision")
        NotebookCard {
            WizardRuler(step = 3, total = 14)
            WizardRuler(step = 12, total = 14)
        }

        SectionHeading("Ledger chart", "direct labels, event rule")
        NotebookCard {
            LedgerChart(
                history = listOf(5.2f, 4.7f, 4.1f, 3.8f, 3.4f, 3.2f, 2.9f, 2.75f),
                annotations = listOf(4 to "checkpoint")
            )
        }

        SectionHeading("Controls", "rectangular, small caps")
        NotebookCard {
            ForgeButton("Start training", onClick = {}, primary = true)
            ForgeOutlineButton("Save preset", onClick = {})
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                StatusTag("stable", ForgeColors.verdigris)
                StatusTag("experimental", ForgeColors.ochreInk)
                StatusTag("planned", ForgeColors.inkMuted)
            }
            LedgerStat("estimated disk", "18.4 MB")
            LedgerStat("measured speed", "81,442 tok/s", valueColor = ForgeColors.emberText)
        }

        SectionHeading("Empty state", "typographic, no illustration")
        NotebookCard {
            TypographicEmptyState(
                "No models on the bench.",
                "Start with a name, a size, and one honest dataset."
            )
        }

        Text(
            "AA audit: run scripts/check_contrast.py. Current pairs pass.",
            style = ForgeType.monoSmall,
            color = ForgeColors.inkMuted,
            modifier = Modifier.padding(bottom = 32.dp)
        )
    }
}

@Composable
private fun RowScope.Swatch(name: String, color: Color, onColor: Color) {
    Column(
        modifier = Modifier
            .weight(1f)
            .background(color, RoundedCornerShape(4.dp))
            .height(64.dp)
            .padding(6.dp),
        verticalArrangement = Arrangement.Bottom
    ) {
        Text(name, style = ForgeType.monoSmall, color = onColor)
    }
}
