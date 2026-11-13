package com.modelforge.app.ui

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import androidx.compose.foundation.background
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.modelforge.app.diagnostics.CheckState
import com.modelforge.app.diagnostics.CrashGuard
import com.modelforge.app.diagnostics.DiagnosticCheck
import com.modelforge.app.diagnostics.DiagnosticsReport
import com.modelforge.app.diagnostics.DeviceReport
import com.modelforge.app.diagnostics.ForgeLog
import com.modelforge.app.diagnostics.LogEntry
import com.modelforge.app.ui.design.ForgeButton
import com.modelforge.app.ui.design.ForgeOutlineButton
import com.modelforge.app.ui.design.Hairline
import com.modelforge.app.ui.design.LedgerStat
import com.modelforge.app.ui.design.NotebookCard
import com.modelforge.app.ui.design.SectionHeading
import com.modelforge.app.ui.design.SmallCaps
import com.modelforge.app.ui.design.StatusTag
import com.modelforge.app.ui.theme.ForgeColors
import com.modelforge.app.ui.theme.ForgeType
import java.text.DateFormat
import java.util.Date

@Composable
fun DiagnosticsScreen(vm: ModelForgeViewModel) {
    val context = LocalContext.current
    val application = context.applicationContext as Application
    val state by vm.diagnostics.collectAsState()
    val logs by ForgeLog.entries.collectAsState()
    var crash by remember { mutableStateOf(CrashGuard.lastCrash(application)) }
    val deviceRows = remember { DeviceReport.rows(context) }

    LaunchedEffect(Unit) {
        if (!state.running && state.lastRun == 0L) vm.runDiagnostics()
    }

    fun report(): String = DiagnosticsReport.build(context, state, logs, crash)

    fun copyReport() {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("ModelForge diagnostics", report()))
    }

    fun shareReport() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "ModelForge diagnostics")
            putExtra(Intent.EXTRA_TEXT, report())
        }
        context.startActivity(Intent.createChooser(intent, "Share diagnostics"))
    }

    ForgeScaffold(
        title = "Diagnostics",
        subtitle = "Local evidence for what this build can do. Secrets are redacted.",
        onBack = vm::goHome,
        actions = {
            ForgeOutlineButton(
                if (state.running) "Checking…" else "Run checks",
                onClick = vm::runDiagnostics,
                enabled = !state.running
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
                    SmallCaps("THIS BUILD")
                    LedgerStat("app", deviceRows.firstOrNull { it.label == "app" }?.value ?: "unknown")
                    LedgerStat("device", deviceRows.firstOrNull { it.label == "device" }?.value ?: "unknown")
                    LedgerStat("Android", deviceRows.firstOrNull { it.label == "Android" }?.value ?: "unknown")
                    LedgerStat("ABI", deviceRows.firstOrNull { it.label == "ABI" }?.value ?: "unknown")
                    LedgerStat("heap", deviceRows.firstOrNull { it.label == "heap limit" }?.value ?: "unknown")
                    Hairline()
                    Text(
                        if (state.lastRun > 0L) {
                            "Last check: ${DateFormat.getDateTimeInstance().format(Date(state.lastRun))}"
                        } else {
                            "Checks have not completed yet."
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = ForgeColors.inkMuted
                    )
                }
            }

            item { SectionHeading("CHECKS", if (state.running) "running" else "explicit") }
            item { CheckCard(state.huggingFace) }
            item { CheckCard(state.nvidia) }
            item { CheckCard(state.secureStorage) }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ForgeButton(
                        "Copy report",
                        onClick = { copyReport() },
                        modifier = Modifier.weight(1f),
                        primary = true
                    )
                    ForgeOutlineButton(
                        "Share",
                        onClick = { shareReport() },
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item { SectionHeading("SYSTEM", "read from this device") }
            item {
                NotebookCard {
                    deviceRows.forEach { row -> LedgerStat(row.label, row.value) }
                }
            }

            if (!crash.isNullOrBlank()) {
                item { SectionHeading("LAST CRASH", "stored locally") }
                item {
                    NotebookCard {
                        Text(
                            crash.orEmpty().take(6_000),
                            style = ForgeType.monoSmall,
                            color = ForgeColors.error
                        )
                        ForgeOutlineButton(
                            "Clear crash report",
                            onClick = {
                                CrashGuard.clear(application)
                                crash = null
                                ForgeLog.info("crash", "Stored crash report cleared")
                            },
                            tint = ForgeColors.error
                        )
                    }
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SmallCaps("RECENT LOG")
                        Text(
                            "${logs.size} entries",
                            style = ForgeType.monoSmall,
                            color = ForgeColors.inkMuted
                        )
                    }
                    ForgeOutlineButton(
                        "Clear log",
                        onClick = {
                            ForgeLog.clear()
                            ForgeLog.info("diagnostics", "Log cleared from Diagnostics")
                        }
                    )
                }
                Hairline()
            }

            if (logs.isEmpty()) {
                item {
                    NotebookCard {
                        Text(
                            "No entries yet.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = ForgeColors.inkMuted
                        )
                    }
                }
            } else {
                items(logs.takeLast(250)) { entry ->
                    LogLine(entry)
                }
            }
        }
    }
}

@Composable
private fun CheckCard(check: DiagnosticCheck) {
    NotebookCard {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                check.name,
                style = MaterialTheme.typography.headlineSmall,
                color = ForgeColors.ink
            )
            StatusTag(check.state.label, checkColor(check.state))
        }
        Text(
            check.detail,
            style = MaterialTheme.typography.bodySmall,
            color = if (check.state == CheckState.FAIL) ForgeColors.error else ForgeColors.inkMuted
        )
    }
}

@Composable
private fun LogLine(entry: LogEntry) {
    Text(
        entry.line(),
        modifier = Modifier
            .fillMaxWidth()
            .background(ForgeColors.inkWash, RoundedCornerShape(3.dp))
            .padding(horizontal = 8.dp, vertical = 5.dp),
        style = ForgeType.monoSmall,
        color = when (entry.level) {
            com.modelforge.app.diagnostics.LogLevel.CRASH,
            com.modelforge.app.diagnostics.LogLevel.ERROR -> ForgeColors.error
            com.modelforge.app.diagnostics.LogLevel.WARN -> ForgeColors.ochreInk
            else -> ForgeColors.inkMuted
        }
    )
}

private fun checkColor(state: CheckState): Color = when (state) {
    CheckState.PASS -> ForgeColors.verdigris
    CheckState.WARN -> ForgeColors.ochreInk
    CheckState.FAIL -> ForgeColors.error
    CheckState.RUNNING -> ForgeColors.inkMuted
    CheckState.INFO -> ForgeColors.inkMuted
}
