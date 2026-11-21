package com.modelforge.app.provider

import android.content.ContentProvider
import android.content.ContentValues
import android.content.UriMatcher
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import com.modelforge.app.data.ModelStore
import com.modelforge.app.training.CheckpointHeader
import java.io.File
import java.io.FileNotFoundException

/**
 * Read-only model surface for ModelForge Bench.
 *
 * Bench copies a model through this provider instead of asking the user to
 * export and re-import gigabytes by hand. Every column and file is derived from
 * the same [ModelStore] the app itself uses, so what Bench measures is exactly
 * what ModelForge would export. Nothing here is writable: [insert], [update]
 * and [delete] all return without changing state.
 */
class ModelsProvider : ContentProvider() {

    private val matcher = UriMatcher(UriMatcher.NO_MATCH).apply {
        addURI(AUTHORITY, "models", MODELS)
        addURI(AUTHORITY, "models/*/*", MODEL_FILE)
    }

    override fun onCreate(): Boolean = true

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?
    ): Cursor {
        if (matcher.match(uri) != MODELS) {
            throw IllegalArgumentException("Only the models collection can be queried: $uri")
        }
        val store = ModelStore(requireContext())
        val cursor = MatrixCursor(COLUMNS)
        store.listManifests().forEach { manifest ->
            val id = manifest.id
            val best = store.bestCheckpoint(id)
            val last = store.lastCheckpoint(id)
            val hasBest = CheckpointHeader.hasMagic(best)
            val hasLast = CheckpointHeader.hasMagic(last)
            val packed = packedFiles(store, id).maxByOrNull { it.second }
            cursor.addRow(
                arrayOf(
                    id,
                    manifest.name,
                    manifest.creator,
                    manifest.createdAt,
                    manifest.config.paramCount(),
                    if (hasBest) 1 else 0,
                    if (hasLast) 1 else 0,
                    if (hasBest) best.length() else 0L,
                    if (hasLast) last.length() else 0L,
                    store.tokenizerFile(id).takeIf { it.isFile }?.length() ?: 0L,
                    packed?.second ?: 0,
                    packed?.first?.length() ?: 0L
                )
            )
        }
        return cursor
    }

    override fun getType(uri: Uri): String = when (matcher.match(uri)) {
        MODELS -> "vnd.android.cursor.dir/vnd.com.modelforge.app.models"
        MODEL_FILE -> "application/octet-stream"
        else -> throw IllegalArgumentException("Unknown URI: $uri")
    }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
        if (mode != "r") throw FileNotFoundException("ModelForge models are read-only.")
        if (matcher.match(uri) != MODEL_FILE) {
            throw FileNotFoundException("No file at $uri")
        }
        val id = uri.pathSegments.getOrNull(1)
            ?: throw FileNotFoundException("Missing model id.")
        val kind = uri.pathSegments.getOrNull(2)
            ?: throw FileNotFoundException("Missing file kind.")
        val store = ModelStore(requireContext())
        // Reject anything that is not a real manifest id, so a crafted URI can
        // never escape the models directory.
        val manifest = store.listManifests().firstOrNull { it.id == id }
            ?: throw FileNotFoundException("Unknown model: $id")
        val file = when (kind) {
            "manifest" -> store.manifestFile(manifest.id)
            "tokenizer" -> store.tokenizerFile(manifest.id)
            "checkpoint" -> {
                when (uri.getQueryParameter("which") ?: "best") {
                    "last" -> store.lastCheckpoint(manifest.id)
                    "recovery" -> store.recoveryCheckpoint(manifest.id)
                    else -> store.bestCheckpoint(manifest.id)
                }
            }
            "packed" -> {
                val bits = uri.getQueryParameter("bits")?.toIntOrNull()
                val candidates = packedFiles(store, manifest.id)
                if (bits != null) {
                    store.packedFile(manifest.id, bits)
                } else {
                    candidates.maxByOrNull { it.second }?.first
                }
            }
            "corpus" -> {
                val index = uri.getQueryParameter("index")?.toIntOrNull() ?: 0
                store.cacheFileFor(manifest.id, index)
            }
            else -> null
        } ?: throw FileNotFoundException("Unknown file kind: $kind")
        if (!file.isFile) throw FileNotFoundException("${file.name} does not exist yet.")
        if (kind == "checkpoint" && !CheckpointHeader.hasMagic(file)) {
            throw FileNotFoundException("${file.name} is not a complete checkpoint.")
        }
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?
    ): Int = 0

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

    private fun packedFiles(store: ModelStore, id: String): List<Pair<File, Int>> {
        val dir = store.packedDir(id)
        return dir.listFiles { file -> file.isFile && file.extension == "mfpk" }
            ?.mapNotNull { file ->
                val bits = file.nameWithoutExtension.substringAfterLast("_q").toIntOrNull()
                    ?: return@mapNotNull null
                file to bits
            }
            ?: emptyList()
    }

    companion object {
        const val AUTHORITY = "com.modelforge.app.models"
        const val PERMISSION = "com.modelforge.app.permission.READ_MODELS"
        private const val MODELS = 1
        private const val MODEL_FILE = 2

        /** Column names are part of the Bench contract; do not rename them. */
        val COLUMNS = arrayOf(
            "id",
            "name",
            "creator",
            "created",
            "params",
            "has_best",
            "has_last",
            "best_bytes",
            "last_bytes",
            "tokenizer_bytes",
            "packed_bits",
            "packed_bytes"
        )
    }
}
