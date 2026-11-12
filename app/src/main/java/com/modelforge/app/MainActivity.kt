package com.modelforge.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.modelforge.app.core.ConfigDeriver
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.ui.theme.ForgeAccent
import com.modelforge.app.ui.theme.ForgeMuted
import com.modelforge.app.ui.theme.ModelForgeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ModelForgeTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    PhaseOneScreen()
                }
            }
        }
    }
}

@Composable
private fun PhaseOneScreen() {
    val context = LocalContext.current.applicationContext
    val hw = remember { HardwareDetector.detect(context) }
    val derived = remember { ConfigDeriver.derive(hw.maxTrainableParams) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("ModelForge", style = MaterialTheme.typography.headlineLarge, color = ForgeAccent)
        Text(
            "Train a small language model entirely on this phone.",
            style = MaterialTheme.typography.bodyLarge,
            color = ForgeMuted
        )
        Spacer(Modifier.height(16.dp))
        Text("Device", style = MaterialTheme.typography.titleMedium)
        Text(hw.summary(), style = MaterialTheme.typography.bodyMedium)
        Text("Battery: ${if (hw.batteryPercent >= 0) "${hw.batteryPercent}%" else "unknown"}")
        Text("Thermal: ${hw.thermalStatus}")
        Spacer(Modifier.height(12.dp))
        Text("Largest model this device can train", style = MaterialTheme.typography.titleMedium)
        Text(
            ConfigDeriver.formatCount(hw.maxTrainableParams) + " parameters",
            style = MaterialTheme.typography.headlineSmall,
            color = ForgeAccent
        )
        Text(hw.limitReason, style = MaterialTheme.typography.bodySmall, color = ForgeMuted)
        Spacer(Modifier.height(12.dp))
        Text("Example 1M architecture", style = MaterialTheme.typography.titleMedium)
        Text(derived.explanation, style = MaterialTheme.typography.bodyMedium)
        Text(
            "Counted parameters: ${derived.config.paramCount()} " +
                "(${(derived.errorFraction * 100).format(2)}% from target)",
            style = MaterialTheme.typography.bodyMedium,
            color = ForgeAccent
        )
        Spacer(Modifier.height(24.dp))
        Text(
            "Phase 1: project skeleton, CI, hardware detection and configuration derivation. " +
                "The full wizard, engine and training arrive in later phases.",
            style = MaterialTheme.typography.bodySmall,
            color = ForgeMuted
        )
    }
}

private fun Double.format(decimals: Int): String = String.format("%.${decimals}f", this)
