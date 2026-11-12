package com.modelforge.app.training

import com.modelforge.app.core.ModelConfig
import com.modelforge.app.data.DatasetMixEntry
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingSettings
import com.modelforge.app.data.TrainingStatus
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class TrainingSupportTest {

    @get:Rule
    val temp = TemporaryFolder()

    private val config = ModelConfig(
        vocabSize = 2048,
        contextLength = 256,
        dModel = 256,
        layers = 6,
        heads = 8,
        dFf = 1024
    )

    @Test
    fun lrScheduleWarmsUpThenDecaysToTenPercent() {
        val total = 100
        val warmup = 10
        val base = 1e-3f
        assertEquals(base / warmup, LrSchedule.learningRate(0, total, warmup, base), 1e-9f)
        assertEquals(base, LrSchedule.learningRate(warmup - 1, total, warmup, base), 1e-6f)
        assertEquals(base, LrSchedule.learningRate(warmup, total, warmup, base), 1e-6f)
        assertEquals(base * 0.1f, LrSchedule.learningRate(total, total, warmup, base), 1e-6f)

        var previous = Float.MAX_VALUE
        for (step in warmup..total step 5) {
            val lr = LrSchedule.learningRate(step, total, warmup, base)
            assertTrue("schedule should not increase after warmup at step $step", lr <= previous + 1e-9f)
            previous = lr
        }
    }

    @Test
    fun lrScheduleHandlesZeroAndTinyRuns() {
        assertEquals(1e-3f, LrSchedule.learningRate(3, 0, 10, 1e-3f), 0f)
        val lr = LrSchedule.learningRate(1, 1, 1, 2e-3f)
        assertTrue(lr > 0f)
    }

    @Test
    fun memoryGuardKeepsBatchWhenRamIsPlentiful() {
        val adjustment = MemoryGuard.adjust(config, 8, 128, availableBytes = 8L * 1024 * 1024 * 1024)
        assertEquals(8, adjustment.batchSize)
        assertNull(adjustment.message)
    }

    @Test
    fun memoryGuardLowersBatchAndExplains() {
        val activation = MemoryGuard.activationBytes(config, 8, 256)
        assertTrue("expected large activation footprint, got $activation", activation > 64L * 1024 * 1024)
        val adjustment = MemoryGuard.adjust(config, 8, 256, availableBytes = 260L * 1024 * 1024)
        assertTrue(adjustment.batchSize in 1..7)
        assertNotNull(adjustment.message)
        assertTrue(adjustment.message!!.contains("Close other apps"))
    }

    @Test
    fun activationEstimateGrowsWithBatchAndSequence() {
        val small = MemoryGuard.activationBytes(config, 1, 64)
        val biggerBatch = MemoryGuard.activationBytes(config, 4, 64)
        val longerSeq = MemoryGuard.activationBytes(config, 1, 128)
        assertTrue(biggerBatch > small)
        assertTrue(longerSeq > small)
    }

    @Test
    fun datasetMixerRespectsWeightsAndReportsMissingFiles() {
        val dir = temp.newFolder("data")
        File(dir, "a.txt").writeText("AAAA".repeat(2000))
        File(dir, "b.txt").writeText("BBBB".repeat(2000))
        val entries = listOf(
            entry("a", "a.txt", weight = 3f),
            entry("b", "b.txt", weight = 1f),
            entry("missing", "nope.txt", weight = 1f)
        )
        val corpus = DatasetMixer.buildWeightedCorpus(entries, dir, maxChars = 4000)
        assertEquals(2, corpus.usedFiles)
        assertEquals(listOf("nope.txt"), corpus.missingFiles)
        assertTrue(corpus.text.length <= 4000)
        assertTrue(corpus.text.contains('A'))
        assertTrue(corpus.text.contains('B'))
    }

    @Test
    fun datasetMixerFallsBackToEqualWeightsWhenAllAreZero() {
        val dir = temp.newFolder("data2")
        File(dir, "a.txt").writeText("alpha")
        File(dir, "b.txt").writeText("beta")
        val corpus = DatasetMixer.buildWeightedCorpus(
            listOf(entry("a", "a.txt", 0f), entry("b", "b.txt", 0f)), dir, 100
        )
        assertEquals(2, corpus.usedFiles)
    }

    @Test
    fun tokenCacheRoundTripsAndRejectsStaleSignature() {
        val dir = temp.newFolder("cache")
        val tokenizerFile = File(dir, "tokenizer.json").apply { writeText("[\"t\"]") }
        val entries = listOf(entry("a", "a.txt", 1f))
        File(dir, "a.txt").writeText("hello")
        val signature = TokenCache.signature(tokenizerFile, entries, dir, config.vocabSize)
        val cache = File(dir, "tokens.bin")
        val manifest = File(dir, "tokens_manifest.json")
        val tokens = IntArray(50) { it % 256 }

        TokenCache.save(cache, manifest, tokens, signature, config.vocabSize)
        val loaded = TokenCache.load(cache, manifest, signature)
        assertNotNull(loaded)
        assertTrue(tokens.contentEquals(loaded!!))
        assertNull(TokenCache.load(cache, manifest, signature + "x"))
    }

    @Test
    fun liveStateAndManifestRoundTripThroughJson() {
        val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
        val state = TrainingLiveState(
            modelId = "demo-1",
            status = TrainingStatus.PAUSED,
            message = "Battery low.",
            step = 42,
            totalSteps = 500,
            loss = 1.25f,
            bestLoss = 1.1f,
            tokensPerSec = 1234.5,
            lossHistory = listOf(2f, 1.5f, 1.25f),
            autoPaused = true,
            batchSize = 4
        )
        val restored = json.decodeFromString<TrainingLiveState>(json.encodeToString(state))
        assertEquals(state, restored)

        val manifest = ModelManifest(
            id = "demo-1",
            name = "Demo",
            createdAt = 123L,
            config = config.copy(bits = 4, qat = true),
            settings = TrainingSettings.autoFor(config),
            datasets = listOf(entry("a", "a.txt", 1f)),
            tokenizerTrained = true
        )
        val restoredManifest = json.decodeFromString<ModelManifest>(json.encodeToString(manifest))
        assertEquals(manifest, restoredManifest)
    }

    @Test
    fun trainingBusPublishesLatestState() {
        val state = TrainingLiveState("bus-model", TrainingStatus.RUNNING, step = 7)
        TrainingBus.publish(state)
        assertEquals(state, TrainingBus.peek())
    }

    private fun entry(id: String, cacheFile: String, weight: Float) = DatasetMixEntry(
        datasetId = id,
        config = "default",
        split = "train",
        columns = listOf("text"),
        weight = weight,
        cacheFile = cacheFile
    )
}
