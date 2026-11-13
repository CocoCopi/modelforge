package com.modelforge.app.data.hf

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import retrofit2.HttpException
import java.io.File
import java.io.IOException

interface AiDatasetSource {
    suspend fun search(query: String, limit: Int = 30): List<HfDatasetSummary>
    suspend fun detail(id: String): HfDatasetSummary
    suspend fun structure(datasetId: String): DatasetStructure
}

class HfRepository(private val bundle: HfApiBundle = HfClient.create()) : AiDatasetSource {

    override suspend fun search(query: String, limit: Int): List<HfDatasetSummary> {
        return try {
            bundle.publicApi.searchDatasets(query, limit)
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    override suspend fun detail(id: String): HfDatasetSummary {
        return try {
            val d = bundle.publicApi.datasetDetail(id)
            HfDatasetSummary(
                id = d.id,
                downloads = d.downloads,
                likes = d.likes,
                license = d.license,
                tags = d.tags,
                cardData = d.cardData
            )
        } catch (t: Throwable) {
            throw HfException(describeNetworkError(t), t)
        }
    }

    override suspend fun structure(datasetId: String): DatasetStructure {
        try {
            val info = bundle.serverApi.info(datasetId)
            if (info.dataset_info.isNotEmpty()) {
                val configs = info.dataset_info.map { (name, entry) ->
                    DatasetConfigInfo(
                        name = name,
                        splits = entry.splits.keys.sorted(),
                        textColumns = textColumns(entry.features)
                    )
                }
                if (configs.any { it.textColumns.isNotEmpty() }) {
                    return DatasetStructure(datasetId, configs)
                }
            }
        } catch (_: Throwable) {
            // Fall through to the splits endpoint; some datasets have no info yet.
        }
        return try {
            val splits = bundle.serverApi.splits(datasetId)
            if (splits.splits.isEmpty()) {
                throw HfException("No configs or splits are available for $datasetId.")
            }
            val grouped = splits.splits.groupBy { it.config }
            val configs = grouped.map { (config, entries) ->
                val splitNames = entries.map { it.split }.distinct().sorted()
                DatasetConfigInfo(
                    config,
                    splitNames,
                    textColumnsFromRows(datasetId, config, splitNames)
                )
            }
            DatasetStructure(datasetId, configs)
        } catch (t: Throwable) {
            if (t is HfException) throw t
            throw HfException(describeNetworkError(t), t)
        }
    }

    private suspend fun textColumnsFromRows(
        datasetId: String,
        config: String,
        splits: List<String>
    ): List<String> {
        for (split in splits) {
            val columns = runCatching {
                retryTransient {
                    bundle.serverApi.rows(datasetId, config, split, 0, 1).features
                        .filter { isTextFeature(it) }
                        .map { it.name }
                }
            }.getOrDefault(emptyList())
            if (columns.isNotEmpty()) return columns
        }
        return emptyList()
    }

    suspend fun downloadText(
        datasetId: String,
        config: String,
        split: String,
        columns: List<String>,
        maxRows: Int,
        maxBytes: Long,
        destination: File,
        pageSize: Int = 100,
        onProgress: (DownloadStats) -> Unit = {}
    ): DownloadStats = withContext(Dispatchers.IO) {
        destination.parentFile?.mkdirs()
        var offset = 0L
        var writtenRows = 0
        var writtenBytes = 0L
        val preview = StringBuilder()
        var totalRows = Long.MAX_VALUE

        destination.bufferedWriter(Charsets.UTF_8).use { writer ->
            while (writtenRows < maxRows && writtenBytes < maxBytes && offset < totalRows) {
                currentCoroutineContext().ensureActive()
                val page = try {
                    retryTransient {
                        bundle.serverApi.rows(
                            datasetId,
                            config,
                            split,
                            offset,
                            minOf(pageSize, maxRows - writtenRows)
                        )
                    }
                } catch (t: Throwable) {
                    throw HfException(describeNetworkError(t), t)
                }
                if (page.rows.isEmpty()) break
                totalRows = if (page.num_rows_total > 0) page.num_rows_total else totalRows
                offset += page.rows.size

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
                    if (writtenBytes + bytes > maxBytes && writtenRows > 0) break
                    writer.write(line)
                    writer.newLine()
                    writtenRows++
                    writtenBytes += bytes
                    if (preview.length < 4096) preview.append(line.take(4096 - preview.length)).append('\n')
                    if (writtenRows >= maxRows) break
                }

                val stats = DownloadStats(writtenRows, writtenBytes, preview.toString())
                onProgress(stats)
            }
        }

        if (writtenRows == 0) {
            throw HfException(
                "No text rows were available for $datasetId / $config / $split. " +
                    "Try another config, split or text column."
            )
        }
        DownloadStats(writtenRows, writtenBytes, preview.toString())
    }

    private suspend fun <T> retryTransient(block: suspend () -> T): T {
        var last: Throwable? = null
        repeat(3) { attempt ->
            try {
                return block()
            } catch (t: Throwable) {
                val transient = t is IOException ||
                    (t is HttpException && t.code() in setOf(429, 500, 502, 503, 504))
                if (!transient || attempt == 2) throw t
                last = t
                delay(1_500L * (attempt + 1))
            }
        }
        throw last ?: IllegalStateException("request failed")
    }

    companion object {

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
