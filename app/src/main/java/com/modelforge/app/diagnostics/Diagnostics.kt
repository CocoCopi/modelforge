package com.modelforge.app.diagnostics

import android.content.Context

enum class CheckState(val label: String) {
    INFO("info"),
    RUNNING("running"),
    PASS("pass"),
    WARN("warn"),
    FAIL("fail")
}

data class DiagnosticCheck(
    val name: String,
    val state: CheckState = CheckState.INFO,
    val detail: String = "Not run"
)

data class DiagnosticsState(
    val running: Boolean = false,
    val huggingFace: DiagnosticCheck = DiagnosticCheck(
        "Hugging Face",
        detail = "Not checked"
    ),
    val nvidia: DiagnosticCheck = DiagnosticCheck(
        "NVIDIA",
        detail = "Not checked"
    ),
    val secureStorage: DiagnosticCheck = DiagnosticCheck(
        "Secure storage",
        detail = "Not checked"
    ),
    val lastRun: Long = 0L
)

object DiagnosticsReport {

    fun build(
        context: Context,
        state: DiagnosticsState,
        logs: List<LogEntry>,
        crash: String?
    ): String = buildString {
        appendLine(DeviceReport.text(context))
        appendLine()
        appendLine("checks")
        listOf(state.huggingFace, state.nvidia, state.secureStorage).forEach { check ->
            appendLine("${check.name.padEnd(18)}: ${check.state.label} - ${check.detail}")
        }
        appendLine()
        appendLine("recent log (${logs.size} entries)")
        logs.takeLast(300).forEach { appendLine(it.line()) }
        if (!crash.isNullOrBlank()) {
            appendLine()
            appendLine("last crash")
            appendLine(crash)
        }
    }
}
