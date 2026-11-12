package com.modelforge.app.training

import android.content.Context
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.core.ModelConfig
import com.modelforge.app.data.DatasetMixEntry
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.cos

object LrSchedule {
    /** Linear warmup followed by cosine decay to 10% of the base rate. */
    fun learningRate(step: Int, totalSteps: Int, warmupSteps: Int, base: Float): Float {
        if (totalSteps <= 0) return base
        val s = step.coerceAtLeast(0)
        val warmup = warmupSteps.coerceIn(0, totalSteps)
        if (warmup > 0 && s < warmup) {
            return base * (s + 1).toFloat() / warmup.toFloat()
        }
        val span = (totalSteps - warmup).coerceAtLeast(1)
        val t = ((s - warmup).toDouble() / span).coerceIn(0.0, 1.0)
        return (base * (0.1 + 0.45 * (1.0 + cos(PI * t)))).toFloat()
    }
}

data class BatchAdjustment(val batchSize: Int, val message: String?)

object MemoryGuard {
    private const val RESERVE_BYTES = 192L * 1024L * 1024L

    fun activationBytes(config: ModelConfig, batch: Int, seq: Int): Long {
        if (!config.isValid() || batch <= 0 || seq <= 0) return 0
        val n = batch.toLong() * seq.toLong()
        val d = config.dModel.toLong()
        val df = config.dFf.toLong()
        val v = config.vocabSize.toLong()
        val heads = config.heads.toLong()
        val perLayer = n * (14 * d + 4 * df) + batch.toLong() * heads * seq.toLong() * seq.toLong()
        val floats = perLayer * config.layers + n * (2 * v + 3 * d)
        return (floats * 4L * 1.2).toLong()
    }

    fun adjust(
        config: ModelConfig,
        requestedBatch: Int,
        seq: Int,
        availableBytes: Long,
        reserveBytes: Long = RESERVE_BYTES
    ): BatchAdjustment {
        val requested = requestedBatch.coerceAtLeast(1)
        val usable = (availableBytes - reserveBytes).coerceAtLeast(32L * 1024L * 1024L)
        var batch = requested
        while (batch > 1 && activationBytes(config, batch, seq) > usable) {
            batch /= 2
        }
        if (batch == requested) return BatchAdjustment(batch, null)
        val est = activationBytes(config, batch, seq) / (1024.0 * 1024.0)
        val availableMb = availableBytes / (1024.0 * 1024.0)
        return BatchAdjustment(
            batch,
            "Batch size lowered from $requested to $batch: this model needs about " +
                String.format("%.0f MB", est) + " of activations and only " +
                String.format("%.0f MB", availableMb) + " RAM is currently available. " +
                "Close other apps or choose a smaller model for a larger batch."
        )
    }

    fun adjustForDevice(config: ModelConfig, requestedBatch: Int, seq: Int, context: Context): BatchAdjustment {
        return adjust(config, requestedBatch, seq, HardwareDetector.availableRamBytes(context))
    }
}

data class WeightedCorpus(
    val text: String,
    val usedFiles: Int,
    val missingFiles: List<String>
)

object DatasetMixer {
    fun buildWeightedCorpus(
        entries: List<DatasetMixEntry>,
        dataDir: File,
        maxChars: Int
    ): WeightedCorpus {
        if (entries.isEmpty() || maxChars <= 0) return WeightedCorpus("", 0, emptyList())
        val positive = entries.filter { it.weight > 0f }
        val effective = if (positive.isEmpty()) entries.map { it.copy(weight = 1f) } else positive
        val sum = effective.sumOf { it.weight.toDouble() }.coerceAtLeast(1e-9)
        val out = StringBuilder(maxChars + 1024)
        val missing = ArrayList<String>()
        var used = 0
        for (entry in effective) {
            val file = File(dataDir, entry.cacheFile)
            if (!file.exists() || file.length() == 0L) {
                missing.add(entry.cacheFile)
                continue
            }
            val share = (maxChars.toDouble() * entry.weight / sum).toInt().coerceAtLeast(1)
            val prefix = readPrefix(file, share)
            if (prefix.isNotBlank()) {
                if (out.isNotEmpty()) out.append("\n\n")
                out.append(prefix)
                used++
            }
            if (out.length >= maxChars) break
        }
        val text = if (out.length > maxChars) out.substring(0, maxChars) else out.toString()
        return WeightedCorpus(text, used, missing)
    }

    private fun readPrefix(file: File, maxChars: Int): String {
        if (maxChars <= 0) return ""
        val sb = StringBuilder(minOf(maxChars, 1 shl 20))
        try {
            file.bufferedReader(Charsets.UTF_8).use { reader ->
                val buffer = CharArray(8192)
                while (sb.length < maxChars) {
                    val toRead = minOf(buffer.size, maxChars - sb.length)
                    val read = reader.read(buffer, 0, toRead)
                    if (read <= 0) break
                    sb.append(buffer, 0, read)
                }
            }
        } catch (_: Exception) {
            return sb.toString()
        }
        return sb.toString()
    }
}

@Serializable
data class TokenCacheManifest(
    val signature: String,
    val tokenCount: Int,
    val vocabSize: Int,
    val createdAt: Long = System.currentTimeMillis()
)

object TokenCache {
    private const val MAGIC = 0x4D46544B
    private const val VERSION = 1
    private const val MAX_TOKENS = 20_000_000
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun signature(
        tokenizerFile: File,
        entries: List<DatasetMixEntry>,
        dataDir: File,
        vocabSize: Int
    ): String {
        val md = MessageDigest.getInstance("SHA-256")
        fun mix(text: String) {
            md.update(text.toByteArray(Charsets.UTF_8))
            md.update(0)
        }
        mix("v1")
        mix(vocabSize.toString())
        mix(tokenizerFile.length().toString())
        mix(tokenizerFile.lastModified().toString())
        for (entry in entries.sortedBy { it.cacheFile }) {
            val file = File(dataDir, entry.cacheFile)
            mix(entry.datasetId)
            mix(entry.cacheFile)
            mix(entry.weight.toString())
            mix(file.length().toString())
            mix(file.lastModified().toString())
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun load(cacheFile: File, manifestFile: File, signature: String): IntArray? {
        if (!cacheFile.exists() || !manifestFile.exists()) return null
        val manifest = runCatching {
            json.decodeFromString<TokenCacheManifest>(manifestFile.readText())
        }.getOrNull() ?: return null
        if (manifest.signature != signature) return null
        if (manifest.tokenCount <= 0 || manifest.tokenCount > MAX_TOKENS) return null
        return try {
            DataInputStream(FileInputStream(cacheFile).buffered(1 shl 16)).use { input ->
                if (input.readInt() != MAGIC) return null
                if (input.readInt() != VERSION) return null
                val count = input.readInt()
                if (count != manifest.tokenCount) return null
                val tokens = IntArray(count)
                for (i in 0 until count) tokens[i] = input.readInt()
                tokens
            }
        } catch (_: Exception) {
            null
        }
    }

    fun save(
        cacheFile: File,
        manifestFile: File,
        tokens: IntArray,
        signature: String,
        vocabSize: Int
    ) {
        cacheFile.parentFile?.mkdirs()
        DataOutputStream(FileOutputStream(cacheFile).buffered(1 shl 16)).use { output ->
            output.writeInt(MAGIC)
            output.writeInt(VERSION)
            output.writeInt(tokens.size)
            for (token in tokens) output.writeInt(token)
        }
        manifestFile.writeText(
            json.encodeToString(
                TokenCacheManifest(signature, tokens.size, vocabSize)
            )
        )
    }
}
