package com.modelforge.app.data.tokenizer

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream
import java.io.File

@Serializable
data class TokenizerData(
    val version: Int = 1,
    val merges: List<List<Int>> = emptyList()
)

/**
 * Byte-level BPE. The first 256 ids are raw bytes; every learned merge gets
 * id 256 + mergeIndex. Training is done over whitespace/non-whitespace chunks,
 * which keeps the algorithm fast while preserving the input exactly.
 */
class BpeTokenizer private constructor(
    private val mergePairs: List<Pair<Int, Int>>
) {
    val vocabSize: Int get() = BASE + mergePairs.size

    private val ranks: HashMap<Long, Int> = HashMap<Long, Int>(mergePairs.size * 2).apply {
        mergePairs.forEachIndexed { index, (a, b) -> put(key(a, b), index) }
    }

    private val tokenBytes: Array<ByteArray?> by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        val table = arrayOfNulls<ByteArray>(vocabSize)
        for (i in 0 until 256) table[i] = byteArrayOf(i.toByte())
        mergePairs.forEachIndexed { index, (a, b) ->
            val left = table.getOrNull(a)
            val right = table.getOrNull(b)
            if (left != null && right != null) {
                table[BASE + index] = left + right
            }
        }
        table
    }

    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val out = GrowableInts(text.length)
        for (chunk in splitChunks(text)) {
            val bytes = chunk.toByteArray(Charsets.UTF_8)
            if (bytes.isEmpty()) continue
            var symbols = IntArray(bytes.size) { bytes[it].toInt() and 0xff }
            while (symbols.size > 1) {
                var bestRank = Int.MAX_VALUE
                var bestPair = 0L
                for (i in 0 until symbols.size - 1) {
                    val key = key(symbols[i], symbols[i + 1])
                    val rank = ranks[key] ?: continue
                    if (rank < bestRank) {
                        bestRank = rank
                        bestPair = key
                    }
                }
                if (bestRank == Int.MAX_VALUE) break
                val a = (bestPair ushr 32).toInt()
                val b = (bestPair and 0xffffffffL).toInt()
                val merged = IntArray(symbols.size)
                var count = 0
                var i = 0
                while (i < symbols.size) {
                    if (i + 1 < symbols.size && symbols[i] == a && symbols[i + 1] == b) {
                        merged[count++] = BASE + bestRank
                        i += 2
                    } else {
                        merged[count++] = symbols[i]
                        i++
                    }
                }
                symbols = merged.copyOf(count)
            }
            for (symbol in symbols) out.add(symbol)
        }
        return out.toArray()
    }

    fun decode(ids: IntArray): String {
        if (ids.isEmpty()) return ""
        val bytes = ByteArrayOutputStream(ids.size * 2)
        for (id in ids) {
            if (id < 0) continue
            val token = if (id < 256) byteArrayOf(id.toByte()) else tokenBytes.getOrNull(id)
            if (token != null) bytes.write(token)
        }
        return bytes.toString(Charsets.UTF_8.name())
    }

    fun decode(ids: List<Int>): String = decode(ids.toIntArray())

    fun toJson(): String = json.encodeToString(
        TokenizerData(merges = mergePairs.map { listOf(it.first, it.second) })
    )

    fun save(file: File) {
        file.parentFile?.mkdirs()
        file.writeText(toJson(), Charsets.UTF_8)
    }

    fun compressionRatio(text: String): Double {
        val raw = text.toByteArray(Charsets.UTF_8).size
        if (raw == 0) return 0.0
        return 1.0 - encode(text).size.toDouble() / raw.toDouble()
    }

    override fun toString(): String = "BpeTokenizer(vocab=$vocabSize, merges=${mergePairs.size})"

    companion object {
        const val BASE = 256
        private const val TRAIN_BUDGET = 300_000_000L

        private val json = Json {
            ignoreUnknownKeys = true
            encodeDefaults = true
        }

        fun train(
            text: String,
            targetVocab: Int,
            maxTrainingChars: Int = 0,
            onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }
        ): BpeTokenizer {
            val targetMerges = (targetVocab - BASE).coerceAtLeast(0)
            if (targetMerges == 0 || text.isEmpty()) return BpeTokenizer(emptyList())

            var limit = text.length
            if (maxTrainingChars > 0) limit = minOf(limit, maxTrainingChars)
            val budgetChars = (TRAIN_BUDGET / targetMerges).toInt().coerceIn(20_000, 1_000_000)
            limit = minOf(limit, budgetChars)
            val sample = text.substring(0, limit)

            val frequencies = HashMap<String, Int>()
            for (chunk in splitChunks(sample)) {
                frequencies[chunk] = (frequencies[chunk] ?: 0) + 1
            }
            if (frequencies.isEmpty()) return BpeTokenizer(emptyList())

            val chunks = ArrayList<IntArray>(frequencies.size)
            val weights = IntArray(frequencies.size)
            var ci = 0
            for ((chunk, freq) in frequencies) {
                val bytes = chunk.toByteArray(Charsets.UTF_8)
                val symbols = IntArray(bytes.size) { bytes[it].toInt() and 0xff }
                if (symbols.isNotEmpty()) {
                    chunks.add(symbols)
                    weights[ci] = freq
                    ci++
                }
            }
            val merges = ArrayList<Pair<Int, Int>>(targetMerges)
            val mergedPairs = HashSet<Long>()

            for (iteration in 0 until targetMerges) {
                val counts = HashMap<Long, Int>(chunks.size * 2)
                for (j in chunks.indices) {
                    val chunk = chunks[j]
                    val freq = weights[j]
                    for (i in 0 until chunk.size - 1) {
                        val key = key(chunk[i], chunk[i + 1])
                        counts[key] = (counts[key] ?: 0) + freq
                    }
                }
                var bestKey = 0L
                var bestCount = 1
                for ((key, count) in counts) {
                    if (count > bestCount || (count == bestCount && key < bestKey)) {
                        bestCount = count
                        bestKey = key
                    }
                }
                if (bestCount < 2 || mergedPairs.contains(bestKey)) break

                val a = (bestKey ushr 32).toInt()
                val b = (bestKey and 0xffffffffL).toInt()
                merges.add(a to b)
                mergedPairs.add(bestKey)
                val newId = BASE + merges.size - 1

                for (j in chunks.indices) {
                    val chunk = chunks[j]
                    var found = false
                    for (i in 0 until chunk.size - 1) {
                        if (chunk[i] == a && chunk[i + 1] == b) {
                            found = true
                            break
                        }
                    }
                    if (!found) continue
                    val out = IntArray(chunk.size)
                    var count = 0
                    var i = 0
                    while (i < chunk.size) {
                        if (i + 1 < chunk.size && chunk[i] == a && chunk[i + 1] == b) {
                            out[count++] = newId
                            i += 2
                        } else {
                            out[count++] = chunk[i]
                            i++
                        }
                    }
                    chunks[j] = out.copyOf(count)
                }
                onProgress(iteration + 1, targetMerges)
            }
            return BpeTokenizer(merges)
        }

        fun fromJson(text: String): BpeTokenizer {
            val data = json.decodeFromString<TokenizerData>(text)
            val merges = ArrayList<Pair<Int, Int>>(data.merges.size)
            for ((index, pair) in data.merges.withIndex()) {
                if (pair.size != 2) throw IllegalArgumentException("Malformed merge at index $index")
                val a = pair[0]
                val b = pair[1]
                val nextId = BASE + index
                if (a !in 0 until nextId || b !in 0 until nextId) {
                    throw IllegalArgumentException("Merge $index references an unknown token")
                }
                merges.add(a to b)
            }
            return BpeTokenizer(merges)
        }

        fun load(file: File): BpeTokenizer = fromJson(file.readText(Charsets.UTF_8))

        private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

        private fun splitChunks(text: String): Sequence<String> = sequence {
            if (text.isEmpty()) return@sequence
            var start = 0
            var currentWhitespace = text[0].isWhitespace()
            for (i in 1 until text.length) {
                val ws = text[i].isWhitespace()
                if (ws != currentWhitespace) {
                    yield(text.substring(start, i))
                    start = i
                    currentWhitespace = ws
                }
            }
            yield(text.substring(start))
        }
    }
}

private class GrowableInts(initialCapacity: Int) {
    private var data = IntArray(maxOf(16, initialCapacity))
    private var size = 0

    fun add(value: Int) {
        if (size == data.size) data = data.copyOf(data.size * 2)
        data[size++] = value
    }

    fun toArray(): IntArray = data.copyOf(size)
}
