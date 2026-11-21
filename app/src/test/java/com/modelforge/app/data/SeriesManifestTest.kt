package com.modelforge.app.data

import com.modelforge.app.core.ModelConfig
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class SeriesManifestTest {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    private val config = ModelConfig(
        vocabSize = 512,
        contextLength = 128,
        dModel = 64,
        layers = 4,
        heads = 4,
        dFf = 256
    )

    @Test
    fun legacyManifestsBecomeTheirOwnSeries() {
        val legacy = ModelManifest(
            id = "atlas",
            name = "Atlas",
            createdAt = 1L,
            config = config,
            settings = TrainingSettings()
        )
        val decoded = json.decodeFromString(
            ModelManifest.serializer(),
            json.encodeToString(ModelManifest.serializer(), legacy)
        )
        assertEquals(1, decoded.seriesVersion)
        assertEquals("", decoded.seriesId)
        assertEquals("", decoded.parentId)
        assertEquals("atlas", decoded.lineageId)
    }

    @Test
    fun seriesMetadataSurvivesARoundTrip() {
        val fork = ModelManifest(
            id = "atlas-v2",
            name = "Atlas v2",
            createdAt = 2L,
            config = config,
            settings = TrainingSettings(),
            seriesId = "atlas",
            seriesVersion = 2,
            parentId = "atlas"
        )
        val decoded = json.decodeFromString(
            ModelManifest.serializer(),
            json.encodeToString(ModelManifest.serializer(), fork)
        )
        assertEquals("atlas", decoded.seriesId)
        assertEquals("atlas", decoded.lineageId)
        assertEquals(2, decoded.seriesVersion)
        assertEquals("atlas", decoded.parentId)
    }
}
