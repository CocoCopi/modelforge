package com.modelforge.app.data

import android.content.Context
import com.modelforge.app.core.ArchitectureRegistry
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class ModelPreset(
    val id: String,
    val name: String,
    val architectureId: String = ArchitectureRegistry.GPT_TEXT,
    val targetParams: Long = 5_000_000L,
    val contextLength: Int = 0,
    val languages: List<String> = listOf("English"),
    val speedMode: String = "normal",
    val reasoningEnabled: Boolean = false,
    val reasoningRoom: Int = 256,
    val toolCalling: Boolean = false,
    val persona: String = "",
    val bits: Int = 4,
    val steps: Int = 500,
    val batchSize: Int = 8,
    val seqLength: Int = 128,
    val learningRate: Float = 6e-4f,
    val warmupSteps: Int = 25,
    val checkpointEvery: Int = 50,
    val sampleEvery: Int = 100,
    val corpusChars: Int = 1_500_000,
    val createdAt: Long = System.currentTimeMillis()
) {
    val summary: String
        get() = "${ArchitectureRegistry.info(architectureId).displayName} · " +
            "${targetParams / 1_000_000}M · ${bits}-bit · $steps steps"
}

class PresetStore(context: Context) {
    private val prefs = context.applicationContext
        .getSharedPreferences("model_presets_v1", Context.MODE_PRIVATE)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    fun list(): List<ModelPreset> {
        val text = prefs.getString(KEY, null) ?: return emptyList()
        return runCatching {
            json.decodeFromString<List<ModelPreset>>(text)
        }.getOrElse { emptyList() }
    }

    fun save(preset: ModelPreset): List<ModelPreset> {
        val next = (list().filterNot { it.id == preset.id } + preset)
            .sortedByDescending { it.createdAt }
            .take(MAX_PRESETS)
        prefs.edit().putString(KEY, json.encodeToString(next)).apply()
        return next
    }

    fun delete(id: String): List<ModelPreset> {
        val next = list().filterNot { it.id == id }
        prefs.edit().putString(KEY, json.encodeToString(next)).apply()
        return next
    }

    private companion object {
        const val KEY = "presets"
        const val MAX_PRESETS = 20
    }
}
