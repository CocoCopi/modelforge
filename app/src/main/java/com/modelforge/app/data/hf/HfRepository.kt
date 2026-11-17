package com.modelforge.app.data.hf

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import com.modelforge.app.diagnostics.ForgeLog
import retrofit2.HttpException
import java.io.File
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap

interface AiDatasetSource {
    suspend fun search(query: String, limit: Int = 30): List<HfDatasetSummary>
    suspend fun detail(id: String): HfDatasetSummary
    suspend fun structure(datasetId: String): DatasetStructure
}

class HfRepository(
    private val bundle: HfApiBundle = HfClient.create(),
    private val onThrottle: (String) -> Unit = {}
) : AiDatasetSource {
    private data class CacheEntry<T>(val value: T, val expiresAt: Long)

    private val detailCache = ConcurrentHashMap<String, CacheEntry<HfDatasetSummary>>()
    private val structureCache = ConcurrentHashMap<String, CacheEntry<DatasetStructure>>()

    override suspend fun search(query: String, limit: Int): List<HfDatasetSummary> {
        return try {
            retryTransient { bundle.publicApi.searchDatasets(query, limit) }
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    override suspend fun detail(id: String): HfDatasetSummary {
        cached(detailCache, id)?.let { return it }
        return try {
            val d = retryTransient { bundle.publicApi.datasetDetail(id) }
            val summary = HfDatasetSummary(
                id = d.id,
                downloads = d.downloads,
                likes = d.likes,
                license = d.license,
                tags = d.tags,
                cardData = d.cardData
            )
            detailCache[id] = CacheEntry(summary, System.currentTimeMillis() + CACHE_TTL_MS)
            summary
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    suspend fun validateToken(): String {
        return try {
            val account = retryTransient { bundle.publicApi.whoAmI() }
            val label = account.fullname?.takeIf { it.isNotBlank() } ?: account.name
            if (label.isBlank()) "Credential accepted by Hugging Face." else "Credential accepted as $label."
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    /**
     * Returns the exact number of rows for a split. The datasets-server info
     * endpoint is preferred; if it has no size metadata, one lightweight rows
     * request supplies num_rows_total.
     */
    suspend fun rowCount(datasetId: String, config: String, split: String): Long? {
        val known = structure(datasetId).configs
            .firstOrNull { it.name == config }
            ?.splitRows?.get(split)
            ?.takeIf { it > 0L }
        if (known != null) return known
        return try {
            val page = retryTransient {
                bundle.serverApi.rows(datasetId, config, split, 0, 1)
            }
            page.num_rows_total.takeIf { it > 0L }
        } catch (t: Throwable) {
            if (isRateLimited(t)) throw HfException(describeNetworkError(t), t)
            null
        }
    }

    override suspend fun structure(datasetId: String): DatasetStructure {
        cached(structureCache, datasetId)?.let { return it }
        try {
            val info = retryTransient { bundle.serverApi.info(datasetId) }
            if (info.dataset_info.isNotEmpty()) {
                val configs = info.dataset_info.map { (name, entry) ->
                    DatasetConfigInfo(
                        name = name,
                        splits = entry.splits.keys.sorted(),
                        textColumns = textColumns(entry.features),
                        splitRows = entry.splits.mapValues { it.value.num_examples }
                    )
                }
                if (configs.any { it.textColumns.isNotEmpty() }) {
                    val structure = DatasetStructure(datasetId, configs)
                    structureCache[datasetId] = CacheEntry(structure, System.currentTimeMillis() + CACHE_TTL_MS)
                    return structure
                }
            }
        } catch (t: Throwable) {
            if (isRateLimited(t)) throw HfException(describeNetworkError(t), t)
            // Fall through to the splits endpoint; some datasets have no info yet.
        }
        return try {
            val splits = retryTransient { bundle.serverApi.splits(datasetId) }
            if (splits.splits.isEmpty()) {
                throw HfException("No configs or splits are available for $datasetId.")
            }
            val grouped = splits.splits.groupBy { it.config }
            val configs = grouped.map { (config, entries) ->
                val splitNames = entries.map { it.split }.distinct().sorted()
                DatasetConfigInfo(
                    config,
                    splitNames,
                    textColumnsFromRows(datasetId, config, splitNames),
                    emptyMap()
                )
            }
            val structure = DatasetStructure(datasetId, configs)
            structureCache[datasetId] = CacheEntry(structure, System.currentTimeMillis() + CACHE_TTL_MS)
            structure
        } catch (t: Throwable) {
            if (t is HfException) throw t
            throw HfException(describeNetworkError(t), t)
        }
    }

    private fun <T> cached(map: ConcurrentHashMap<String, CacheEntry<T>>, key: String): T? {
        val entry = map[key] ?: return null
        if (entry.expiresAt <= System.currentTimeMillis()) {
            map.remove(key)
            return null
        }
        return entry.value
    }

    private fun isRateLimited(t: Throwable): Boolean =
        t is HttpException && t.code() == 429

    private suspend fun textColumnsFromRows(
        datasetId: String,
        config: String,
        splits: List<String>
    ): List<String> {
        for (split in splits) {
            try {
                val columns = retryTransient {
                    bundle.serverApi.rows(datasetId, config, split, 0, 1).features
                        .filter { isTextFeature(it) }
                        .map { it.name }
                }
                if (columns.isNotEmpty()) return columns
            } catch (t: Throwable) {
                if (isRateLimited(t)) throw t
            }
        }
        return emptyList()
    }

    suspend fun downloadText(
        datasetId: String,
        config: String,
        split: String,
        columns: List<String>,
        maxRows: Long,
        maxBytes: Long,
        destination: File,
        pageSize: Int = 100,
        concurrency: Int = 12,
        onProgress: (DownloadStats) -> Unit = {}
    ): DownloadStats = withContext(Dispatchers.IO) {
        destination.parentFile?.mkdirs()
        var offset = 0L
        var writtenRows = 0L
        var writtenBytes = 0L
        val preview = StringBuilder()
        var totalRows = Long.MAX_VALUE
        // The datasets-server caps length at 100 rows per page, so the only
        // remaining lever is how many pages are in flight at once.
        val workers = concurrency.coerceIn(1, 16)
        val pageRows = pageSize.coerceIn(1, 100)

        destination.bufferedWriter(Charsets.UTF_8).use { writer ->
            while (writtenRows < maxRows && writtenBytes < maxBytes && offset < totalRows) {
                currentCoroutineContext().ensureActive()
                val stopOffset = minOf(totalRows, maxRows)
                val offsets = (0 until workers)
                    .map { offset + it.toLong() * pageRows }
                    .filter { it < stopOffset }
                if (offsets.isEmpty()) break
                val pages = try {
                    coroutineScope {
                        offsets.map { pageOffset ->
                            val length = minOf(
                                pageRows.toLong(),
                                stopOffset - pageOffset
                            ).toInt().coerceAtLeast(1)
                            async {
                                pageOffset to retryTransient {
                                    bundle.serverApi.rows(
                                        datasetId,
                                        config,
                                        split,
                                        pageOffset,
                                        length
                                    )
                                }
                            }
                        }.awaitAll()
                    }
                } catch (t: Throwable) {
                    throw HfException(describeNetworkError(t), t)
                }

                var consumed = 0L
                var reachedEnd = false
                for ((pageOffset, page) in pages) {
                    if (pageOffset != offset + consumed) break
                    if (page.rows.isEmpty()) {
                        reachedEnd = true
                        break
                    }
                    if (page.num_rows_total > 0) totalRows = page.num_rows_total
                    val pageColumns = if (columns.isNotEmpty()) columns else page.features
                        .filter { isTextFeature(it) }
                        .map { it.name }
                    for (entry in page.rows) {
                        val parts = pageColumns.mapNotNull { col ->
                            (entry.row[col] as? JsonPrimitive)?.contentOrNull?.takeIf { it.isNotBlank() }
                        }
                        if (parts.isEmpty()) continue
                        val line = parts.joinToString("\n")
                        val bytes = line.toByteArray(Charsets.UTF_8).size + 1
                        if (writtenBytes + bytes > maxBytes && writtenRows > 0) {
                            reachedEnd = true
                            break
                        }
                        writer.write(line)
                        writer.newLine()
                        writtenRows++
                        writtenBytes += bytes
                        if (preview.length < 4096) preview.append(line.take(4096 - preview.length)).append('\n')
                        if (writtenRows >= maxRows) {
                            reachedEnd = true
                            break
                        }
                    }
                    consumed += page.rows.size
                    if (reachedEnd) break
                }
                offset += consumed
                val stats = DownloadStats(writtenRows, writtenBytes, preview.toString())
                onProgress(stats)
                if (reachedEnd || consumed == 0L || offset >= totalRows) break
            }
        }

        if (writtenRows == 0L) {
            throw HfException(
                "No text rows were available for $datasetId / $config / $split. " +
                    "Try another config, split or text column."
            )
        }
        DownloadStats(writtenRows, writtenBytes, preview.toString())
    }

    private suspend fun <T> retryTransient(block: suspend () -> T): T {
        var attempt = 0
        while (true) {
            try {
                return block()
            } catch (t: Throwable) {
                val code = (t as? HttpException)?.code()
                val rateLimited = code == 429
                val transient = t is IOException || rateLimited ||
                    code == 500 || code == 502 || code == 503 || code == 504
                val maxAttempts = if (rateLimited) 4 else 3
                if (!transient || attempt >= maxAttempts - 1) throw t
                val retryAfter = (t as? HttpException)?.response()?.headers()
                    ?.get("Retry-After")?.toLongOrNull()
                val seconds = when {
                    rateLimited -> retryAfter?.coerceIn(1L, 120L) ?: RATE_LIMIT_WAIT_SECONDS[attempt]
                    else -> retryAfter?.coerceIn(1L, 30L) ?: (2L * (attempt + 1))
                }
                val notice = if (rateLimited) {
                    "Hugging Face rate limit; waiting ${seconds}s before retrying"
                } else {
                    "Hugging Face is busy; retrying in ${seconds}s"
                }
                ForgeLog.warn("hf", notice)
                onThrottle(notice)
                delay(seconds * 1_000L)
                attempt++
            }
        }
    }

    companion object {
        private const val CACHE_TTL_MS = 10 * 60 * 1_000L
        private val RATE_LIMIT_WAIT_SECONDS = longArrayOf(5, 20, 60)

        fun textColumns(features: Map<String, JsonObject>): List<String> {
            return features.entries
                .filter { (_, value) ->
                    val type = (value["_type"] as? JsonPrimitive)?.contentOrNull
                    val dtype = (value["dtype"] as? JsonPrimitive)?.contentOrNull
                    type == "Value" && dtype == "string"
                }
                .map { it.key }
                .sorted()
        }

        fun isTextFeature(feature: RowFeature): Boolean {
            val dtype = (feature.type["dtype"] as? JsonPrimitive)?.contentOrNull
            val type = (feature.type["_type"] as? JsonPrimitive)?.contentOrNull
            return dtype == "string" && (type == null || type == "Value")
        }
    }
}
