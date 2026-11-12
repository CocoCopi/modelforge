package com.modelforge.app.data.hf

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull

@Serializable
data class HfCardData(
    val license: JsonElement? = null
)

private fun JsonElement?.licenseString(): String? = when (this) {
    is JsonPrimitive -> contentOrNull?.takeIf { it.isNotBlank() }
    is JsonArray -> firstOrNull()?.let { (it as? JsonPrimitive)?.contentOrNull?.takeIf { s -> s.isNotBlank() } }
    else -> null
}

@Serializable
data class HfDatasetSummary(
    val id: String = "",
    val downloads: Long = 0L,
    val likes: Int = 0,
    val license: JsonElement? = null,
    val tags: List<String> = emptyList(),
    val cardData: HfCardData? = null
) {
    fun resolvedLicense(): String? {
        val direct = license.licenseString()
        if (direct != null) return direct
        val card = cardData?.license.licenseString()
        if (card != null) return card
        return tags.firstOrNull { it.startsWith("license:") }
            ?.removePrefix("license:")
            ?.takeIf { it.isNotBlank() }
    }
}

@Serializable
data class HfDatasetDetail(
    val id: String = "",
    val downloads: Long = 0L,
    val likes: Int = 0,
    val license: JsonElement? = null,
    val tags: List<String> = emptyList(),
    val cardData: HfCardData? = null
) {
    fun resolvedLicense(): String? {
        val direct = license.licenseString()
        if (direct != null) return direct
        val card = cardData?.license.licenseString()
        if (card != null) return card
        return tags.firstOrNull { it.startsWith("license:") }
            ?.removePrefix("license:")
            ?.takeIf { it.isNotBlank() }
    }
}

@Serializable
data class DatasetInfoResponse(
    val dataset_info: Map<String, DatasetInfoEntry> = emptyMap()
)

@Serializable
data class DatasetInfoEntry(
    val features: Map<String, JsonObject> = emptyMap(),
    val splits: Map<String, SplitMeta> = emptyMap()
)

@Serializable
data class SplitMeta(
    val num_examples: Long = 0L,
    val num_bytes: Long = 0L
)

@Serializable
data class SplitsResponse(
    val splits: List<SplitEntry> = emptyList(),
    val pending: List<JsonObject> = emptyList(),
    val failed: List<JsonObject> = emptyList()
)

@Serializable
data class SplitEntry(
    val dataset: String = "",
    val config: String = "",
    val split: String = ""
)

@Serializable
data class RowsResponse(
    val features: List<RowFeature> = emptyList(),
    val rows: List<RowEntry> = emptyList(),
    val num_rows_total: Long = 0L,
    val num_rows_per_page: Int = 0,
    val partial: Boolean = false
)

@Serializable
data class RowFeature(
    val feature_idx: Int = 0,
    val name: String = "",
    val type: JsonObject = JsonObject(emptyMap())
)

@Serializable
data class RowEntry(
    val row_idx: Long = 0L,
    val row: JsonObject = JsonObject(emptyMap()),
    val truncated_cells: List<String> = emptyList()
)

data class DatasetConfigInfo(
    val name: String,
    val splits: List<String>,
    val textColumns: List<String>
)

data class DatasetStructure(
    val datasetId: String,
    val configs: List<DatasetConfigInfo>
)

data class DownloadStats(
    val rows: Int,
    val bytes: Long,
    val preview: String
)

class HfException(message: String, cause: Throwable? = null) : Exception(message, cause)

fun isLikelyNonCommercial(license: String?): Boolean {
    if (license.isNullOrBlank()) return true
    val l = license.lowercase()
    return l.contains("non-commercial") ||
        l.contains("noncommercial") ||
        Regex("(^|[-_ ])nc([-_ ]|$)").containsMatchIn(l) ||
        l.contains("research") ||
        l.contains("unknown")
}

fun describeNetworkError(t: Throwable): String {
    val msg = t.message ?: t.javaClass.simpleName
    return when {
        msg.contains("401") -> "Hugging Face rejected the credential (HTTP 401). Check the token."
        msg.contains("403") -> "Hugging Face denied access (HTTP 403). The dataset may be gated; add a valid token."
        msg.contains("404") -> "Dataset, config or split was not found (HTTP 404)."
        msg.contains("429") -> "Hugging Face rate limit reached (HTTP 429). Wait a minute and retry."
        msg.contains("timeout", ignoreCase = true) -> "The request timed out. Check the connection and retry."
        else -> "Network error: $msg"
    }
}
