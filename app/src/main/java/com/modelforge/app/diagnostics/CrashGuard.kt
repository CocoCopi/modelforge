package com.modelforge.app.diagnostics

import android.app.Application
import android.os.Build
import android.os.Process
import com.modelforge.app.BuildConfig
import java.io.File
import java.time.Instant

/**
 * Stores the last uncaught exception on disk so the Diagnostics screen still
 * has it after the process is restarted. The report is local only and is
 * redacted before it is written.
 */
object CrashGuard {

    private const val FILE_NAME = "last-crash.txt"

    fun install(application: Application) {
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            runCatching { write(application, thread, throwable) }
            runCatching {
                ForgeLog.crash(
                    "crash",
                    "Uncaught exception on ${thread.name}",
                    throwable
                )
            }
            if (previous != null) {
                previous.uncaughtException(thread, throwable)
            } else {
                Process.killProcess(Process.myPid())
            }
        }
    }

    private fun write(application: Application, thread: Thread, throwable: Throwable) {
        val report = buildString {
            appendLine("time: ${Instant.now()}")
            appendLine("app: ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
            appendLine("device: ${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("android: ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
            appendLine("abi: ${Build.SUPPORTED_ABIS.joinToString(", ")}")
            appendLine("thread: ${thread.name}")
            appendLine()
            appendLine(SecretRedactor.redact(throwable.stackTraceToString()))
        }
        file(application).writeText(report)
    }

    fun lastCrash(application: Application): String? {
        val file = file(application)
        return if (file.exists()) runCatching { file.readText() }.getOrNull() else null
    }

    fun clear(application: Application) {
        runCatching { file(application).delete() }
    }

    private fun file(application: Application): File {
        val directory = File(application.filesDir, "diagnostics")
        directory.mkdirs()
        return File(directory, FILE_NAME)
    }
}
