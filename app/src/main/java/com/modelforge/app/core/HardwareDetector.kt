package com.modelforge.app.core

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import java.io.File

data class HardwareInfo(
    val totalRamBytes: Long,
    val availableRamBytes: Long,
    val memoryClassMb: Int,
    val isLowRamDevice: Boolean,
    val cpuCores: Int,
    val maxCpuFreqKhz: Long?,
    val batteryPercent: Int,
    val thermalStatus: String,
    val maxTrainableParams: Long,
    val limitReason: String
) {
    fun summary(): String = buildString {
        append("${formatRam(totalRamBytes)} RAM")
        append(" • ${cpuCores} cores")
        maxCpuFreqKhz?.let { append(" • up to ${it / 1000} MHz") }
        append(" • ${isLowRamDevice.let { if (it) "low-RAM device" else "heap ${memoryClassMb} MB" }}")
    }

    companion object {
        fun formatRam(bytes: Long): String = when {
            bytes >= 1L shl 30 -> String.format("%.1f GB", bytes / (1L shl 30).toDouble())
            else -> String.format("%.0f MB", bytes / (1L shl 20).toDouble())
        }
    }
}

object HardwareDetector {

    private const val BYTES_PER_TRAIN_PARAM = 24L
    private const val ABSOLUTE_MAX_PARAMS = 200_000_000L
    private const val MIN_PARAMS = 1_000_000L

    fun detect(context: Context): HardwareInfo {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val totalRam = if (mi.totalMem > 0) mi.totalMem else fallbackTotalRam()
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)

        val usableForTraining = (totalRam * 0.32).toLong()
        val ramCap = usableForTraining / BYTES_PER_TRAIN_PARAM
        val cpuCap = cores.toLong() * 16_000_000L
        val cap = minOf(ramCap, cpuCap, ABSOLUTE_MAX_PARAMS).coerceAtLeast(MIN_PARAMS)

        val reason = when (cap) {
            ramCap -> "Limited by RAM: ${HardwareInfo.formatRam(totalRam)} total, " +
                "about ${HardwareInfo.formatRam(usableForTraining)} can be used safely for training " +
                "(~$BYTES_PER_TRAIN_PARAM bytes per parameter for weights, gradients and AdamW state)."
            cpuCap -> "Limited by CPU: $cores cores; a phone-class CPU can realistically train " +
                "about ${ConfigDeriver.formatCount(cpuCap)} parameters in a reasonable time."
            else -> "Capped at ${ConfigDeriver.formatCount(ABSOLUTE_MAX_PARAMS)} parameters by ModelForge " +
                "for a usable phone-sized model."
        }

        val battery = batteryPercent(context)
        val thermal = thermalStatus(context)

        return HardwareInfo(
            totalRamBytes = totalRam,
            availableRamBytes = mi.availMem,
            memoryClassMb = am.memoryClass,
            isLowRamDevice = am.isLowRamDevice,
            cpuCores = cores,
            maxCpuFreqKhz = maxCpuFreqKhz(),
            batteryPercent = battery,
            thermalStatus = thermal,
            maxTrainableParams = cap,
            limitReason = reason
        )
    }

    fun batteryPercent(context: Context): Int {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
        if (bm != null) {
            val p = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            if (p in 0..100) return p
        }
        val intent = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        ) ?: return -1
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        return if (level >= 0 && scale > 0) level * 100 / scale else -1
    }

    fun thermalStatus(context: Context): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return "unknown (Android < 10)"
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return "unknown"
        return when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "nominal"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "unknown"
        }
    }

    fun shouldPauseForHeat(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
        val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return false
        return pm.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
    }

    private fun fallbackTotalRam(): Long {
        return try {
            val lines = File("/proc/meminfo").readLines()
            val kb = lines.firstOrNull { it.startsWith("MemTotal:") }
                ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull() ?: 0L
            kb * 1024L
        } catch (_: Exception) {
            0L
        }
    }

    private fun maxCpuFreqKhz(): Long? {
        return try {
            val f = File("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
            if (f.exists()) f.readText().trim().toLongOrNull() else null
        } catch (_: Exception) {
            null
        }
    }
}
