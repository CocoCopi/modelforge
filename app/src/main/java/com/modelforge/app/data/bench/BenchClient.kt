package com.modelforge.app.data.bench

import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri

/**
 * Reads ModelForge Bench's run history through its read-only provider.
 *
 * Everything is optional: if Bench is not installed, is older than 1.0.1, or
 * the user has not granted the permission, the series view simply shows no
 * scores instead of pretending a model was evaluated.
 */
class BenchClient(private val context: Context) {

    data class RunSummary(
        val runId: String,
        val benchModelId: String,
        val modelName: String,
        val sourceId: String,
        val modelSha256: String,
        val finishedAt: Long,
        val mode: String,
        val status: String,
        val composite: Double,
        val grade: String,
        val mcAccuracy: Double,
        val perplexity: Double,
        val decodeTps: Double,
        val costPerMillion: Double,
        val weakestPart: String,
        val weakestScore: Double,
        val verdict: String,
        val durationMs: Long
    ) {
        val completed: Boolean get() = status == "COMPLETED" && composite >= 0.0
    }

    fun isInstalled(): Boolean = runCatching {
        context.packageManager.getPackageInfo(PACKAGE, 0)
        true
    }.getOrElse { false }

    fun hasPermission(): Boolean =
        context.checkSelfPermission(PERMISSION) == PackageManager.PERMISSION_GRANTED

    fun availability(): Availability = when {
        !isInstalled() -> Availability.NOT_INSTALLED
        !hasPermission() -> Availability.NEEDS_PERMISSION
        else -> try {
            runs()
            Availability.READY
        } catch (_: SecurityException) {
            Availability.NEEDS_PERMISSION
        } catch (_: IllegalArgumentException) {
            Availability.NOT_SUPPORTED
        } catch (_: Exception) {
            Availability.ERROR
        }
    }

    fun runs(): List<RunSummary> {
        val cursor = context.contentResolver.query(
            Uri.parse("content://$AUTHORITY/runs"), null, null, null, null
        ) ?: return emptyList()
        cursor.use {
            val out = ArrayList<RunSummary>(it.count)
            while (it.moveToNext()) {
                out.add(
                    RunSummary(
                        runId = it.string("id"),
                        benchModelId = it.string("model_id"),
                        modelName = it.string("model_name"),
                        sourceId = it.string("source_id"),
                        modelSha256 = it.string("model_sha256"),
                        finishedAt = it.long("finished_at"),
                        mode = it.string("mode"),
                        status = it.string("status"),
                        composite = it.double("composite"),
                        grade = it.string("grade"),
                        mcAccuracy = it.double("mc_accuracy"),
                        perplexity = it.double("perplexity"),
                        decodeTps = it.double("decode_tps"),
                        costPerMillion = it.double("cost_per_million"),
                        weakestPart = it.string("weakest_part"),
                        weakestScore = it.double("weakest_score"),
                        verdict = it.string("verdict"),
                        durationMs = it.long("duration_ms")
                    )
                )
            }
            return out
        }
    }

    /** Latest completed score for each ModelForge model id. */
    fun latestBySource(): Map<String, RunSummary> =
        runs().filter { it.completed && it.sourceId.isNotBlank() }
            .sortedByDescending { it.finishedAt }
            .groupBy { it.sourceId }
            .mapValues { (_, list) -> list.first() }

    private fun android.database.Cursor.string(name: String): String {
        val index = getColumnIndex(name)
        return if (index >= 0 && !isNull(index)) getString(index) else ""
    }

    private fun android.database.Cursor.long(name: String): Long {
        val index = getColumnIndex(name)
        return if (index >= 0 && !isNull(index)) getLong(index) else 0L
    }

    private fun android.database.Cursor.double(name: String): Double {
        val index = getColumnIndex(name)
        return if (index >= 0 && !isNull(index)) getDouble(index) else -1.0
    }

    enum class Availability { READY, NOT_INSTALLED, NEEDS_PERMISSION, NOT_SUPPORTED, ERROR }

    companion object {
        const val PACKAGE = "com.modelforge.bench"
        const val AUTHORITY = "com.modelforge.bench.runs"
        const val PERMISSION = "com.modelforge.bench.permission.READ_RUNS"
    }
}
