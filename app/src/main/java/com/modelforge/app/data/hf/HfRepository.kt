package com.modelforge.app.data.hf

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import com.modelforge.app.diagnostics.ForgeLog
import retrofit2.HttpException
import java.io.File
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
    private val fileSource: HfFileSource? = bundle.fileApi?.let { api ->
        RetrofitFileSource(api, bundle.authenticated)
    }

    private fun authenticated(): Boolean = bundle.authenticated()

    override suspend fun search(query: String, limit: Int): List<HfDatasetSummary> {
        return try {
            retryTransient(HfRateLimiter.HUB) {
                bundle.publicApi.searchDatasets(query, limit)
            }
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    override suspend fun detail(id: String): HfDatasetSummary {
        cached(detailCache, id)?.let { return it }
        return try {
            val d = retryTransient(HfRateLimiter.HUB) { bundle.publicApi.datasetDetail(id) }
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
            val account = retryTransient(HfRateLimiter.HUB) { bundle.publicApi.whoAmI() }
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
            val page = retryTransient(HfRateLimiter.VIEWER) {
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
            val info = retryTransient(HfRateLimiter.VIEWER) { bundle.serverApi.info(datasetId) }
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
            val splits = retryTransient(HfRateLimiter.VIEWER) { bundle.serverApi.splits(datasetId) }
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
                val columns = retryTransient(HfRateLimiter.VIEWER) {
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
        concurrency: Int = 24,
        knownTotalRows: Long = 0L,
        onProgress: (DownloadStats) -> Unit = {}
    ): DownloadStats = withContext(Dispatchers.IO) {
        destination.parentFile?.mkdirs()
        // Prefer the repository's own data files. This is one or two resolver
        // requests instead of one viewer request per 100 rows, so it is both
        // faster and outside the Hub API quota. Any problem falls through to
        // the viewer, which keeps the old behaviour working.
        val source = fileSource
        if (source != null) {
            val direct = try {
                HfTextFiles.download(
                    source = source,
                    datasetId = datasetId,
                    config = config,
                    split = split,
                    columns = columns,
                    maxRows = maxRows,
                    maxBytes = maxBytes,
                    destination = destination,
                    onProgress = onProgress
                )
            } catch (t: Throwable) {
                if (t is kotlinx.coroutines.CancellationException) throw t
                runCatching { destination.delete() }
                ForgeLog.warn("hf", "Direct download failed for $datasetId; using the viewer", t)
                null
            }
            if (direct != null) return@withContext direct
        }
        var writtenRows = 0L
        var writtenBytes = 0L
        val preview = StringBuilder()
        var totalRows = if (knownTotalRows > 0L) knownTotalRows else Long.MAX_VALUE
        // The datasets-server caps length at 100 rows per page, so the only
        // remaining levers are how many pages are in flight and how well the
        // fetch and write phases overlap. The window below keeps `workers`
        // requests outstanding at all times instead of waiting for a whole
        // batch to finish before starting the next one.
        val workers = concurrency.coerceIn(1, 64)
        val pageRows = pageSize.coerceIn(1, 100)
        val rowLimit = maxRows.coerceAtLeast(1L)

        destination.bufferedWriter(Charsets.UTF_8).use { writer ->
            coroutineScope {
                // Each entry remembers the offset and the length that were
                // requested so the writer can detect a gap even if the server
                // returns fewer rows than asked for.
                val window =
                    ArrayDeque<Triple<Long, Int, kotlinx.coroutines.Deferred<RowsResponse>>>()
                var nextFetch = 0L
                var consumedOffset = 0L
                var finished = false

                suspend fun fillWindow() {
                    val stop = minOf(totalRows, rowLimit)
                    while (!finished && window.size < workers && nextFetch < stop) {
                        val offset = nextFetch
                        val length = minOf(pageRows.toLong(), stop - offset)
                            .toInt().coerceAtLeast(1)
                        val deferred = async {
                            retryTransient(HfRateLimiter.VIEWER) {
                                bundle.serverApi.rows(
                                    datasetId,
                                    config,
                                    split,
                                    offset,
                                    length
                                )
                            }
                        }
                        window.addLast(Triple(offset, length, deferred))
                        nextFetch += length
                    }
                }

                try {
                    fillWindow()
                    while (!finished && window.isNotEmpty()) {
                        currentCoroutineContext().ensureActive()
                        val (pageOffset, pageLength, deferred) = window.removeFirst()
                        val page = try {
                            deferred.await()
                        } catch (t: Throwable) {
                            throw HfException(describeNetworkError(t), t)
                        }
                        if (page.num_rows_total > 0L) {
                            totalRows = page.num_rows_total
                        }
                        // Refill immediately so the next page is already in
                        // flight while this one is written to disk.
                        fillWindow()

                        if (page.rows.isEmpty()) {
                            finished = true
                            break
                        }
                        val pageColumns = if (columns.isNotEmpty()) {
                            columns
                        } else {
                            page.features.filter { isTextFeature(it) }.map { it.name }
                        }
                        if (pageColumns.isEmpty()) {
                            throw HfException(
                                "This split has no text columns. Choose a text column " +
                                    "or another config."
                            )
                        }
                        if (pageOffset != consumedOffset) {
                            // The server returned a page out of order; stop
                            // rather than interleave two ranges.
                            finished = true
                            break
                        }
                        consumedOffset += pageLength
                        for (entry in page.rows) {
                            val parts = pageColumns.mapNotNull { col ->
                                (entry.row[col] as? JsonPrimitive)?.contentOrNull
                                    ?.takeIf { it.isNotBlank() }
                            }
                            if (parts.isEmpty()) continue
                            val line = parts.joinToString("\n")
                            val bytes = line.toByteArray(Charsets.UTF_8).size + 1
                            if (writtenBytes + bytes > maxBytes && writtenRows > 0L) {
                                finished = true
                                break
                            }
                            writer.write(line)
                            writer.newLine()
                            writtenRows++
                            writtenBytes += bytes
                            if (preview.length < 4096) {
                                preview.append(line.take(4096 - preview.length)).append('\n')
                            }
                            if (writtenRows >= rowLimit) {
                                finished = true
                                break
                            }
                        }
                        onProgress(DownloadStats(writtenRows, writtenBytes, preview.toString()))
                    }
                } finally {
                    window.forEach { (_, _, deferred) -> deferred.cancel() }
                }
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

    private suspend fun <T> retryTransient(host: String, block: suspend () -> T): T =
        pacedRequest(
            host = host,
            authenticated = authenticated(),
            maxAttempts = 3,
            onThrottle = { notice ->
                ForgeLog.warn("hf", notice)
                onThrottle(notice)
            },
            block = block
        )

    companion object {
        private const val CACHE_TTL_MS = 10 * 60 * 1_000L

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
