package com.modelforge.app.data

import android.content.Context
import com.modelforge.app.core.ModelConfig
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
private data class ActiveTraining(val modelId: String)

class ModelStore(context: Context) {

    private val root = File(context.applicationContext.filesDir, "models")
    private val activeFile = File(context.applicationContext.filesDir, "active_training.json")

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        prettyPrint = true
    }

    fun modelsRoot(): File {
        root.mkdirs()
        return root
    }

    fun modelDir(id: String): File = File(modelsRoot(), sanitize(id)).apply { mkdirs() }

    fun configFile(id: String): File = File(modelDir(id), "config.json")
    fun manifestFile(id: String): File = File(modelDir(id), "model.json")
    fun tokenizerFile(id: String): File = File(modelDir(id), "tokenizer.json")
    fun checkpointsDir(id: String): File = File(modelDir(id), "checkpoints").apply { mkdirs() }
    fun lastCheckpoint(id: String): File = File(checkpointsDir(id), "last.mfck")
    fun bestCheckpoint(id: String): File = File(checkpointsDir(id), "best.mfck")
    fun recoveryCheckpoint(id: String): File = File(checkpointsDir(id), "recovery.mfck")
    fun packedDir(id: String): File = File(modelDir(id), "packed").apply { mkdirs() }
    fun packedFile(id: String, bits: Int): File = File(packedDir(id), "model_q$bits.mfpk")
    fun dataDir(id: String): File = File(modelDir(id), "data").apply { mkdirs() }
    fun tokenCacheFile(id: String): File = File(dataDir(id), "tokens.bin")
    fun tokenCacheManifestFile(id: String): File = File(dataDir(id), "tokens_manifest.json")
    fun liveStateFile(id: String): File = File(modelDir(id), "training_state.json")
    fun modelCardFile(id: String): File = File(modelDir(id), "MODEL_CARD.md")

    fun cacheFileFor(id: String, index: Int): File = File(dataDir(id), "dataset_$index.txt")

    fun newModelId(name: String): String {
        val slug = name.lowercase()
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .ifBlank { "model" }
            .take(32)
        return "$slug-${System.currentTimeMillis()}"
    }

    fun createModel(manifest: ModelManifest): ModelManifest {
        saveManifest(manifest)
        return manifest
    }

    fun saveManifest(manifest: ModelManifest) {
        val dir = modelDir(manifest.id)
        dir.mkdirs()
        writeAtomic(manifestFile(manifest.id), json.encodeToString(manifest))
        writeAtomic(configFile(manifest.id), json.encodeToString(manifest.config))
    }

    fun loadManifest(id: String): ModelManifest? {
        val file = manifestFile(id)
        if (!file.exists()) return null
        return runCatching { json.decodeFromString<ModelManifest>(file.readText()) }.getOrNull()
    }

    fun loadConfig(id: String): ModelConfig? {
        val file = configFile(id)
        if (!file.exists()) return null
        return runCatching { json.decodeFromString<ModelConfig>(file.readText()) }.getOrNull()
    }

    fun listManifests(): List<ModelManifest> {
        val dirs = modelsRoot().listFiles { f -> f.isDirectory } ?: return emptyList()
        return dirs.mapNotNull { loadManifest(it.name) }.sortedByDescending { it.createdAt }
    }

    fun deleteModel(id: String) {
        modelDir(id).deleteRecursively()
        if (activeModelId() == id) setActive(id, false)
    }

    fun saveLiveState(state: TrainingLiveState) {
        writeAtomic(liveStateFile(state.modelId), json.encodeToString(state))
    }

    fun loadLiveState(id: String): TrainingLiveState? {
        val file = liveStateFile(id)
        if (!file.exists()) return null
        return runCatching { json.decodeFromString<TrainingLiveState>(file.readText()) }.getOrNull()
    }

    fun setActive(modelId: String, active: Boolean) {
        if (active) {
            activeFile.parentFile?.mkdirs()
            writeAtomic(activeFile, json.encodeToString(ActiveTraining(modelId)))
        } else {
            if (activeModelId() == modelId) activeFile.delete()
        }
    }

    fun activeModelId(): String? {
        if (!activeFile.exists()) return null
        return runCatching {
            json.decodeFromString<ActiveTraining>(activeFile.readText()).modelId
        }.getOrNull()
    }

    private fun sanitize(id: String): String =
        id.replace(Regex("[^A-Za-z0-9._-]"), "_").ifBlank { "model" }

    private fun writeAtomic(target: File, text: String) {
        target.parentFile?.mkdirs()
        val tmp = File(target.parentFile, target.name + ".tmp")
        tmp.writeText(text, Charsets.UTF_8)
        if (!tmp.renameTo(target)) {
            target.delete()
            tmp.renameTo(target)
        }
    }
}
