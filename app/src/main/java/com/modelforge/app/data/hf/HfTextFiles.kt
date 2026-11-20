package com.modelforge.app.data.hf

import com.modelforge.app.diagnostics.ForgeLog
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader
import java.io.Reader
import java.util.concurrent.ConcurrentHashMap
import java.util.zip.GZIPInputStream

/**
 * Access to the repository's raw data files. This is deliberately separate from
 * [HfFileApi] so the download logic can be tested without a network stack.
 */
interface HfFileSource {
    suspend fun list(datasetId: String, path: String, revision: String = "main"): List<HfTreeEntry>
    /** Opens at most [maxBytes] of a file, from its first byte. */
    suspend fun open(
        datasetId: String,
        path: String,
        maxBytes: Long,
        revision: String = "main"
    ): InputStream
}

class RetrofitFileSource(
    private val api: HfFileApi,
    private val authenticated: () -> Boolean = { false }
) : HfFileSource {

    private data class Cached(val entries: List<HfTreeEntry>, val expiresAt: Long)

    private val cache = ConcurrentHashMap<String, Cached>()

    override suspend fun list(
        datasetId: String,
        path: String,
        revision: String
    ): List<HfTreeEntry> {
        val key = "$datasetId@$revision/$path"
        cache[key]?.let { if (it.expiresAt > System.currentTimeMillis()) return it.entries }
        val entries = pacedRequest(HfRateLimiter.HUB, authenticated()) {
            if (path.isBlank()) {
                api.treeRoot(datasetId, revision)
            } else {
                api.treeDir(datasetId, revision, path)
            }
        }
        cache[key] = Cached(entries, System.currentTimeMillis() + CACHE_TTL_MS)
        return entries
    }

    override suspend fun open(
        datasetId: String,
        path: String,
        maxBytes: Long,
        revision: String
    ): InputStream {
        val end = maxBytes.coerceAtLeast(1L) - 1L
        val body = pacedRequest(HfRateLimiter.RESOLVER, authenticated()) {
            api.file(HfFileApi.fileUrl(datasetId, revision, path), "bytes=0-$end")
        }
        return body.byteStream()
    }

    private companion object {
        const val CACHE_TTL_MS = 10 * 60 * 1_000L
    }
}

/**
 * Downloads a split from the repository's own JSONL/CSV/TXT/JSON files instead
 * of the dataset viewer.
 *
 * The viewer's `/rows` endpoint returns 100 rows per call and is explicitly
 * meant for previewing, not bulk download. Reading the real data files through
 * resolver URLs turns a 1,000-request download into one or two requests, which
 * is both faster and far away from the Hub's rate limit.
 *
 * Every failure is reported as `null` so the caller can fall back to the viewer;
 * this path never fails a download that the viewer could have completed.
 */
object HfTextFiles {

    private val textExtensions = setOf("jsonl", "json", "csv", "tsv", "txt")

    /** Extensions that are data files from the Hub's point of view. */
    fun isTextPath(path: String): Boolean {
        val name = path.substringAfterLast('/').lowercase()
        if (name == "readme.md" || name == "dataset_infos.json" || name.endsWith(".yaml")) return false
        val base = name.removeSuffix(".gz")
        val ext = base.substringAfterLast('.', "")
        return ext in textExtensions
    }

    private fun splitTokens(split: String): List<String> = when (split.lowercase()) {
        "validation", "valid", "val" -> listOf("validation", "valid", "val")
        "train" -> listOf("train")
        "test" -> listOf("test")
        else -> listOf(split.lowercase())
    }

    private fun hasToken(path: String, tokens: List<String>): Boolean = tokens.any { token ->
        Regex(
            "(?<![a-z0-9_])" + Regex.escape(token) + "(?![a-z0-9_])",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(path)
    }

    /**
     * Picks the files that belong to [split]. A file name that names a
     * different split is never mixed in, because train and test text must not
     * be blended.
     */
    fun candidates(entries: List<HfTreeEntry>, split: String): List<HfTreeEntry> {
        val files = entries.filter { entry ->
            entry.type == "file" && entry.path.isNotBlank() && isTextPath(entry.path) &&
                // A tiny JSON file is metadata (dataset_infos, config), not data.
                !(entry.path.lowercase().endsWith(".json") && entry.size in 1..4_095L)
        }
        if (files.isEmpty()) return emptyList()
        val tokens = splitTokens(split)
        val matching = files.filter { hasToken(it.path, tokens) }
        val chosen = if (matching.isNotEmpty()) {
            matching
        } else {
            val otherSplits = listOf("train", "test", "validation")
                .flatMap { splitTokens(it) }
                .toSet() - tokens.toSet()
            files.filter { file -> !hasToken(file.path, otherSplits.toList()) }
        }
        return chosen.sortedWith(
            compareBy<HfTreeEntry> { extensionRank(it.path) }.thenBy { it.path }
        )
    }

    private fun extensionRank(path: String): Int {
        val name = path.substringAfterLast('/').lowercase().removeSuffix(".gz")
        return when {
            name.endsWith(".jsonl") -> 0
            name.endsWith(".txt") -> 1
            name.endsWith(".csv") -> 2
            name.endsWith(".tsv") -> 3
            else -> 4
        }
    }

    suspend fun download(
        source: HfFileSource,
        datasetId: String,
        config: String,
        split: String,
        columns: List<String>,
        maxRows: Long,
        maxBytes: Long,
        destination: File,
        onProgress: (DownloadStats) -> Unit = {}
    ): DownloadStats? {
        val listings = mutableListOf<HfTreeEntry>()
        val root = try {
            source.list(datasetId, "")
        } catch (t: Throwable) {
            if (t is kotlinx.coroutines.CancellationException) throw t
            ForgeLog.warn("hf", "Could not list $datasetId for a direct download", t)
            return null
        }
        listings += root

        if (candidates(root, split).isEmpty()) {
            // Data often lives one level down, under `data/` or a config
            // directory. Two extra listings at most; the rest is guesswork.
            val directories = root.filter { it.type == "directory" }.map { it.path }
            val wanted = buildList {
                if ("data" in directories) add("data")
                if (config.isNotBlank() && config != "data" && config != "default" &&
                    config in directories
                ) {
                    add(config)
                }
            }
            for (directory in wanted.take(2)) {
                val entries = try {
                    source.list(datasetId, directory)
                } catch (t: Throwable) {
                    if (t is kotlinx.coroutines.CancellationException) throw t
                    continue
                }
                listings += entries
                if (candidates(entries, split).isNotEmpty()) break
            }
        }

        val files = candidates(listings, split)
        if (files.isEmpty()) return null

        destination.parentFile?.mkdirs()
        var writtenRows = 0L
        var writtenBytes = 0L
        val preview = StringBuilder()
        val rowLimit = maxRows.coerceAtLeast(1L)
        val byteLimit = maxBytes.coerceAtLeast(1L)

        val writer = destination.bufferedWriter(Charsets.UTF_8)
        try {
            for (file in files) {
                if (writtenRows >= rowLimit || writtenBytes >= byteLimit) break
                val budget = (byteLimit - writtenBytes).coerceAtLeast(1L)
                try {
                    source.open(datasetId, file.path, budget).use { stream ->
                        forEachRow(stream, file.path, columns) { row ->
                            val bytes = row.toByteArray(Charsets.UTF_8).size + 1
                            if (writtenBytes + bytes > byteLimit && writtenRows > 0L) {
                                return@forEachRow false
                            }
                            writer.write(row)
                            writer.newLine()
                            writtenRows++
                            writtenBytes += bytes
                            if (preview.length < PREVIEW_CHARS) {
                                preview.append(row.take(PREVIEW_CHARS - preview.length)).append('\n')
                            }
                            if (writtenRows % 250L == 0L) {
                                onProgress(
                                    DownloadStats(writtenRows, writtenBytes, preview.toString())
                                )
                            }
                            writtenRows < rowLimit
                        }
                    }
                } catch (t: Throwable) {
                    if (t is CancellationException) throw t
                    // A gzip slice cut at the byte budget ends mid-member, which
                    // is expected. Keep the rows that were already parsed.
                    if (writtenRows == 0L) {
                        ForgeLog.warn("hf", "Direct read of ${file.path} failed", t)
                        continue
                    }
                    break
                }
            }
        } finally {
            writer.close()
        }

        if (writtenRows == 0L) {
            destination.delete()
            return null
        }
        val stats = DownloadStats(writtenRows, writtenBytes, preview.toString())
        onProgress(stats)
        ForgeLog.info(
            "hf",
            "Direct download: $datasetId $split from ${files.size} file(s), " +
                "$writtenRows rows without the 100-row page API"
        )
        return stats
    }

    // --- parsing -----------------------------------------------------------

    /**
     * Streams one text record per line (or one JSON object for `.json`) into
     * [onRow] until it returns false. Returns normally on malformed input so a
     * partially readable file still contributes data.
     */
    fun forEachRow(
        input: InputStream,
        path: String,
        columns: List<String>,
        onRow: (String) -> Boolean
    ) {
        val name = path.lowercase()
        val stream: InputStream = if (name.endsWith(".gz")) {
            GZIPInputStream(input, 1 shl 16)
        } else {
            input
        }
        when {
            name.endsWith(".jsonl") || name.endsWith(".jsonl.gz") ->
                jsonLines(stream, columns, onRow)
            name.endsWith(".csv") || name.endsWith(".csv.gz") ->
                delimited(BufferedReader(InputStreamReader(stream, Charsets.UTF_8)), ',', columns, onRow)
            name.endsWith(".tsv") || name.endsWith(".tsv.gz") ->
                delimited(BufferedReader(InputStreamReader(stream, Charsets.UTF_8)), '\t', columns, onRow)
            name.endsWith(".json") || name.endsWith(".json.gz") ->
                jsonObjects(stream, columns, onRow)
            else -> plainLines(stream, onRow)
        }
    }

    private fun jsonLines(
        stream: InputStream,
        columns: List<String>,
        onRow: (String) -> Boolean
    ) {
        BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 16).use { reader ->
            while (true) {
                val line = reader.readLine() ?: break
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed == "[" || trimmed == "]") continue
                val obj = runCatching {
                    HfClient.json.parseToJsonElement(trimmed) as? JsonObject
                }.getOrNull() ?: continue
                val text = extract(obj, columns) ?: continue
                if (!onRow(text)) break
            }
        }
    }

    private fun plainLines(stream: InputStream, onRow: (String) -> Boolean) {
        BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 16).use { reader ->
            val document = StringBuilder()
            fun flush(): Boolean {
                val text = document.toString().trim()
                document.setLength(0)
                if (text.isEmpty()) return true
                return onRow(text)
            }
            while (true) {
                val line = reader.readLine() ?: break
                val trimmed = line.trim()
                // `<|endoftext|>` is the document separator used by corpora
                // such as TinyStories; a blank line is the generic equivalent.
                if (trimmed == "<|endoftext|>" || trimmed.isEmpty()) {
                    if (!flush()) break
                    continue
                }
                if (document.isNotEmpty()) document.append('\n')
                document.append(trimmed)
                if (document.length >= MAX_DOCUMENT_CHARS && !flush()) break
            }
            if (document.isNotEmpty()) flush()
        }
    }

    /** Reads a pretty-printed JSON array (or one object) without loading it all. */
    private fun jsonObjects(
        stream: InputStream,
        columns: List<String>,
        onRow: (String) -> Boolean
    ) {
        val reader = BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 1 shl 16)
        val buffer = StringBuilder()
        val chunk = CharArray(1 shl 13)
        var depth = 0
        var inString = false
        var escaped = false
        var started = false
        loop@ while (true) {
            val read = reader.read(chunk)
            if (read < 0) break
            for (index in 0 until read) {
                val c = chunk[index]
                if (inString) {
                    if (escaped) {
                        escaped = false
                    } else if (c == '\\') {
                        escaped = true
                    } else if (c == '"') {
                        inString = false
                    }
                    if (depth > 0) buffer.append(c)
                    continue
                }
                when (c) {
                    '"' -> {
                        inString = true
                        if (depth > 0) buffer.append(c)
                    }
                    '{' -> {
                        depth++
                        if (depth == 1) {
                            started = true
                            buffer.setLength(0)
                        }
                        buffer.append(c)
                    }
                    '}' -> {
                        if (depth > 0) buffer.append(c)
                        depth--
                        if (depth == 0 && started) {
                            started = false
                            val obj = runCatching {
                                HfClient.json.parseToJsonElement(buffer.toString()) as? JsonObject
                            }.getOrNull()
                            if (obj != null) {
                                val text = extract(obj, columns)
                                if (text != null && !onRow(text)) break@loop
                            }
                            buffer.setLength(0)
                        }
                    }
                    else -> if (depth > 0) buffer.append(c)
                }
            }
        }
    }

    private fun delimited(
        reader: BufferedReader,
        delimiter: Char,
        columns: List<String>,
        onRow: (String) -> Boolean
    ) {
        reader.use {
            val header = readRecord(it, delimiter) ?: return
            val normalized = header.map { cell ->
                cell.removePrefix("\uFEFF").trim().trim('"').lowercase()
            }
            val indices = if (columns.isEmpty()) {
                normalized.indices.toList()
            } else {
                columns.mapNotNull { column ->
                    normalized.indexOf(column.trim().lowercase()).takeIf { index -> index >= 0 }
                }
            }
            if (indices.isEmpty()) return
            while (true) {
                val record = readRecord(it, delimiter) ?: break
                val parts = indices.mapNotNull { index ->
                    record.getOrNull(index)?.takeIf { cell -> cell.isNotBlank() }
                }
                if (parts.isEmpty()) continue
                if (!onRow(parts.joinToString("\n"))) break
            }
        }
    }

    /** One CSV/TSV record; a quoted field may contain delimiters and newlines. */
    private fun readRecord(reader: Reader, delimiter: Char): List<String>? {
        val fields = mutableListOf<String>()
        val field = StringBuilder()
        var inQuotes = false
        var sawAny = false
        while (true) {
            val value = reader.read()
            if (value < 0) {
                if (!sawAny && field.isEmpty() && fields.isEmpty()) return null
                fields.add(field.toString())
                return fields
            }
            val c = value.toChar()
            sawAny = true
            if (inQuotes) {
                if (c != '"') {
                    field.append(c)
                    continue
                }
                val next = reader.read()
                when {
                    next == '"'.code -> field.append('"')
                    next == delimiter.code -> {
                        fields.add(field.toString())
                        field.setLength(0)
                        inQuotes = false
                    }
                    next == '\n'.code || next < 0 -> {
                        fields.add(field.toString())
                        return fields
                    }
                    next == '\r'.code -> Unit
                    else -> {
                        field.append(next.toChar())
                        inQuotes = false
                    }
                }
                continue
            }
            when (c) {
                '"' -> if (field.isEmpty()) inQuotes = true else field.append(c)
                delimiter -> {
                    fields.add(field.toString())
                    field.setLength(0)
                }
                '\n' -> {
                    fields.add(field.toString())
                    return fields
                }
                '\r' -> Unit
                else -> field.append(c)
            }
        }
    }

    private fun extract(obj: JsonObject, columns: List<String>): String? {
        val values = if (columns.isNotEmpty()) {
            columns.mapNotNull { column ->
                (obj[column] as? JsonPrimitive)
                    ?.takeIf { it.isString }
                    ?.contentOrNull
                    ?.takeIf { it.isNotBlank() }
            }
        } else {
            obj.values.mapNotNull { value ->
                (value as? JsonPrimitive)
                    ?.takeIf { it.isString }
                    ?.contentOrNull
                    ?.takeIf { it.isNotBlank() }
            }
        }
        if (values.isEmpty()) return null
        return values.joinToString("\n")
    }

    private const val PREVIEW_CHARS = 4_096
    private const val MAX_DOCUMENT_CHARS = 64_000
}
