package com.modelforge.app.diagnostics

import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

enum class LogLevel(val label: String) {
    DEBUG("debug"),
    INFO("info"),
    WARN("warn"),
    ERROR("error"),
    CRASH("crash")
}

data class LogEntry(
    val timestamp: Long,
    val level: LogLevel,
    val tag: String,
    val message: String
) {
    fun timeText(): String = TIME_FORMAT.format(
        Instant.ofEpochMilli(timestamp).atZone(ZoneId.systemDefault())
    )

    fun line(): String = "${timeText()} ${level.label.uppercase()} $tag: $message"

    companion object {
        private val TIME_FORMAT = DateTimeFormatter.ofPattern("HH:mm:ss.SSS")
    }
}

/**
 * Structured in-process log with a bounded ring buffer. Credentials are
 * redacted before an entry is stored or written to Logcat. The buffer is
 * intentionally not uploaded anywhere.
 */
object ForgeLog {

    private const val MAX_ENTRIES = 1_000
    private const val LOGCAT_TAG = "ModelForge"
    private val buffer = ArrayDeque<LogEntry>()
    private val lock = Any()
    private val _entries = MutableStateFlow<List<LogEntry>>(emptyList())
    val entries: StateFlow<List<LogEntry>> = _entries.asStateFlow()

    fun debug(tag: String, message: String) = append(LogLevel.DEBUG, tag, message, null)

    fun info(tag: String, message: String) = append(LogLevel.INFO, tag, message, null)

    fun warn(tag: String, message: String, error: Throwable? = null) =
        append(LogLevel.WARN, tag, message, error)

    fun error(tag: String, message: String, error: Throwable? = null) =
        append(LogLevel.ERROR, tag, message, error)

    fun crash(tag: String, message: String, error: Throwable? = null) =
        append(LogLevel.CRASH, tag, message, error)

    private fun append(level: LogLevel, tag: String, message: String, error: Throwable?) {
        val safeTag = SecretRedactor.redact(tag).take(80)
        val detail = buildString {
            append(SecretRedactor.redact(message))
            if (error != null) {
                append("\n")
                append(SecretRedactor.redact(error.stackTraceToString()))
            }
        }
        runCatching {
            val priority = when (level) {
                LogLevel.DEBUG -> Log.DEBUG
                LogLevel.INFO -> Log.INFO
                LogLevel.WARN -> Log.WARN
                LogLevel.ERROR, LogLevel.CRASH -> Log.ERROR
            }
            Log.println(priority, "$LOGCAT_TAG/$safeTag", detail)
        }
        val entry = LogEntry(
            timestamp = System.currentTimeMillis(),
            level = level,
            tag = safeTag,
            message = detail
        )
        synchronized(lock) {
            buffer.addLast(entry)
            while (buffer.size > MAX_ENTRIES) buffer.removeFirst()
            _entries.value = buffer.toList()
        }
    }

    fun clear() {
        synchronized(lock) {
            buffer.clear()
            _entries.value = emptyList()
        }
    }

    fun exportText(): String = synchronized(lock) {
        buffer.joinToString(separator = "\n", postfix = if (buffer.isEmpty()) "" else "\n") { it.line() }
    }
}
