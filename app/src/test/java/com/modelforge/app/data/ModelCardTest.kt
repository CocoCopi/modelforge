package com.modelforge.app.data

import com.modelforge.app.core.ArchitectureRegistry
import com.modelforge.app.core.ModelConfig
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelCardTest {
    private fun manifest() = ModelManifest(
        id = "kite-1",
        name = "Kite",
        creator = "Rin",
        createdAt = 1_700_000_000_000L,
        config = ModelConfig(
            layers = 4,
            dModel = 128,
            heads = 4,
            dFf = 512,
            vocabSize = 4096,
            contextLength = 128,
            bits = 4,
            architectureId = ArchitectureRegistry.GPT_TEXT
        ),
        settings = TrainingSettings(steps = 500, identityFraction = 0.04f),
        datasets = listOf(
            DatasetMixEntry(
                datasetId = "tiny",
                config = "default",
                split = "train",
                columns = listOf("text"),
                weight = 1f,
                cacheFile = "dataset_0.txt",
                rows = 1000,
                license = "mit"
            )
        ),
        languages = listOf("English", "Spanish"),
        behavior = BehaviorConfig(reasoningEnabled = true, reasoningRoom = 384)
    )

    @Test
    fun card_names_the_model_creator_and_licence() {
        val card = ModelCard.build(manifest(), generatedAt = 1_700_000_000_000L)
        assertTrue(card.contains("# Kite"))
        assertTrue(card.contains("Created by Rin"))
        assertTrue(card.contains("licence mit"))
        assertTrue(card.contains("English, Spanish"))
        assertTrue(card.contains("Identity fraction: 4.0%"))
        assertTrue(card.contains("You are responsible"))
        assertTrue(!card.contains("state-of-the-art"))
    }
}
