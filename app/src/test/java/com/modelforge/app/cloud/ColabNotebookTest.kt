package com.modelforge.app.cloud

import com.modelforge.app.core.ModelConfig
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.TrainingSettings
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ColabNotebookTest {

    private fun manifest() = ModelManifest(
        id = "atlas-123",
        name = "Atlas v2",
        creator = "Test",
        createdAt = 1L,
        config = ModelConfig(
            vocabSize = 512,
            contextLength = 128,
            dModel = 64,
            layers = 4,
            heads = 4,
            dFf = 256
        ),
        settings = TrainingSettings(
            steps = 1234,
            batchSize = 6,
            seqLength = 96,
            learningRate = 3e-4f,
            warmupSteps = 40,
            checkpointEvery = 25
        ),
        seriesId = "atlas",
        seriesVersion = 2
    )

    @Test
    fun notebookCarriesTheModelAndItsTrainingSettings() {
        val text = ColabNotebook.notebook(manifest())
        assertTrue(text.contains("\"nbformat\": 4"))
        assertTrue(text.contains("Atlas v2"))
        assertTrue(text.contains("ModelForge/atlas-123"))
        assertTrue(text.contains("str(STEPS)"))
        assertTrue(text.contains("STEPS = 1234"))
        assertTrue(text.contains("'--batch', '6'"))
        assertTrue(text.contains("'--seq', '96'"))
        assertTrue(text.contains("'--warmup', '40'"))
        assertTrue(text.contains("'--every', '25'"))
        assertTrue(text.contains(ColabNotebook.REPO_URL))
        assertTrue(text.trim().startsWith("{"))
        assertTrue(text.trim().endsWith("}"))
    }

    @Test
    fun notebookIsHonestAboutRuntimesAndContainsNoCredentials() {
        val text = ColabNotebook.notebook(manifest())
        assertTrue(text.contains("CPU-only"))
        assertTrue(text.contains("output/best.mfck"))
        assertFalse(text.contains("AIza"))
        assertFalse(text.contains("access_token"))
        assertFalse(text.contains("refresh_token"))
        assertFalse(text.contains("colab-cli"))
    }

    @Test
    fun everyCellHasTheFieldsColabExpects() {
        val text = ColabNotebook.notebook(manifest())
        // 1 markdown + 6 code cells.
        assertTrue(text.contains("\"cell_type\": \"markdown\""))
        assertEquals(6, Regex("\"cell_type\": \"code\"").findAll(text).count())
        assertEquals(6, Regex("\"execution_count\": null").findAll(text).count())
        assertEquals(6, Regex("\"outputs\": \\[\\]").findAll(text).count())
    }

    @Test
    fun notebookParsesAsRealJson() {
        val element = kotlinx.serialization.json.Json.parseToJsonElement(
            ColabNotebook.notebook(manifest())
        )
        val root = element as kotlinx.serialization.json.JsonObject
        val cells = root.getValue("cells") as kotlinx.serialization.json.JsonArray
        org.junit.Assert.assertEquals(7, cells.size)
        org.junit.Assert.assertEquals(4, root.getValue("nbformat").toString().toInt())
    }

    private fun assertEquals(expected: Int, actual: Int) {
        org.junit.Assert.assertEquals(expected, actual)
    }
}
