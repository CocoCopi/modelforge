package com.modelforge.app.data

import android.content.Context
import com.modelforge.app.core.ArchitectureRegistry
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

@Serializable
data class WizardDraft(
    val modelId: String = "",
    val name: String = "",
    val creator: String = "",
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
    val datasets: List<DatasetMixEntry> = emptyList(),
    val step: Int = 0
)

class WizardDraftStore(context: Context) {
    private val prefs = context.applicationContext
        .getSharedPreferences("wizard_draft_v2", Context.MODE_PRIVATE)
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    fun load(): WizardDraft? {
        val text = prefs.getString(KEY, null) ?: return null
        return runCatching { json.decodeFromString<WizardDraft>(text) }.getOrNull()
    }

    fun save(draft: WizardDraft) {
        prefs.edit().putString(KEY, json.encodeToString(draft)).apply()
    }

    fun clear() {
        prefs.edit().remove(KEY).apply()
    }

    private companion object {
        const val KEY = "draft"
    }
}
