package com.modelforge.app.diagnostics

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs
import com.modelforge.app.BuildConfig
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.core.HardwareInfo
import java.util.Locale

object DeviceReport {

    data class Row(val label: String, val value: String)

    fun rows(context: Context): List<Row> {
        val packageInfo = runCatching {
            context.packageManager.getPackageInfo(context.packageName, 0)
        }.getOrNull()
        val versionName = packageInfo?.versionName ?: BuildConfig.VERSION_NAME
        val versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo?.longVersionCode?.toString()
        } else {
            @Suppress("DEPRECATION")
            packageInfo?.versionCode?.toString()
        } ?: BuildConfig.VERSION_CODE.toString()

        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memory = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memory)
        val stat = runCatching { StatFs(context.filesDir.absolutePath) }.getOrNull()
        val hardware = runCatching { HardwareDetector.detect(context) }.getOrNull()
        val runtime = Runtime.getRuntime()

        return buildList {
            add(Row("app", "$versionName ($versionCode)"))
            add(Row("package", context.packageName))
            add(Row("device", "${Build.MANUFACTURER} ${Build.MODEL}"))
            add(Row("Android", "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})"))
            add(Row("ABI", Build.SUPPORTED_ABIS.joinToString(", ")))
            add(Row("CPU cores", Runtime.getRuntime().availableProcessors().toString()))
            add(Row("RAM", HardwareInfo.formatRam(memory.totalMem)))
            add(Row("RAM available", HardwareInfo.formatRam(memory.availMem)))
            add(Row("heap limit", "${runtime.maxMemory() / (1024L * 1024L)} MB"))
            add(Row("heap used", "${(runtime.totalMemory() - runtime.freeMemory()) / (1024L * 1024L)} MB"))
            stat?.let {
                add(Row("storage free", HardwareInfo.formatRam(it.availableBytes)))
                add(Row("storage total", HardwareInfo.formatRam(it.totalBytes)))
            }
            hardware?.let {
                add(Row("battery", if (it.batteryPercent >= 0) "${it.batteryPercent}%" else "unknown"))
                add(Row("thermal", it.thermalStatus))
                add(Row("training ceiling", "${it.maxTrainableParams} params (physical estimate)"))
            }
            add(Row("locale", Locale.getDefault().toLanguageTag()))
        }
    }

    fun text(context: Context): String = buildString {
        appendLine("ModelForge diagnostics")
        rows(context).forEach { row ->
            append(row.label.padEnd(18))
            append(": ")
            appendLine(row.value)
        }
    }
}
