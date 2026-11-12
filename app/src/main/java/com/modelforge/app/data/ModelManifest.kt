package com.modelforge.app.data

import com.modelforge.app.core.ModelConfig
import kotlinx.serialization.Serializable

@Serializable
data class DatasetMixEntry(
    val datasetId: String,
    val config: String,
    val split: String,
    val columns: List<String>,
    val weight: Float,
    val cacheFile: String,
    val rows: Int = 0,
    val bytes: Long = 0L,
    val license: String? = null,
    val nonCommercial: Boolean = false
)

@Serializable
data class TrainingSettings(
    val steps: Int = 500,
    val batchSize: Int = 8,
    val learningRate: Float = 6e-4f,
    val warmupSteps: Int = 25,
    val seqLength: Int = 128,
    val weightDecay: Float = 0.01f,
    val checkpointEvery: Int = 50,
    val sampleEvery: Int = 100,
    val samplingTemperature: Float = 0.8f,
    val corpusChars: Int = 1_500_000,
    val autoResume: Boolean = true,
    val identityFraction: Float = 0.04f
) {
    companion object {
        fun autoFor(config: ModelConfig): TrainingSettings {
            val params = config.paramCount()
            val seq = when {
                params <= 2_000_000L -> minOf(config.contextLength, 128)
                params <= 15_000_000L -> minOf(config.contextLength, 192)
                else -> minOf(config.contextLength, 256)
            }.coerceAtLeast(32)
            val batch = when {
                params <= 2_000_000L -> 8
                params <= 15_000_000L -> 6
                else -> 4
            }
            val steps = when {
                params <= 2_000_000L -> 500
                params <= 15_000_000L -> 300
                else -> 200
            }
            val lr = when {
                params <= 2_000_000L -> 1e-3f
                params <= 15_000_000L -> 6e-4f
                else -> 3e-4f
            }
            return TrainingSettings(
                steps = steps,
                batchSize = batch,
                learningRate = lr,
                warmupSteps = maxOf(10, steps / 20),
                seqLength = seq,
                checkpointEvery = maxOf(20, steps / 10),
                sampleEvery = maxOf(20, steps / 5),
                corpusChars = (steps.toLong() * batch * seq * 3L)
                    .coerceIn(200_000L, 2_000_000L)
                    .toInt()
            )
        }
    }
}

@Serializable
data class BehaviorConfig(
    val speedMode: String = "normal",
    val reasoningEnabled: Boolean = false,
    val reasoningRoom: Int = 256,
    val toolCalling: Boolean = false,
    val persona: String = ""
)

@Serializable
data class ModelManifest(
    val id: String,
    val name: String,
    val creator: String = "",
    val createdAt: Long,
    val config: ModelConfig,
    val settings: TrainingSettings,
    val datasets: List<DatasetMixEntry> = emptyList(),
    val tokenizerTrained: Boolean = false,
    val languages: List<String> = emptyList(),
    val behavior: BehaviorConfig = BehaviorConfig()
)

@Serializable
data class TrainingLiveState(
    val modelId: String,
    val status: String,
    val message: String = "",
    val step: Int = 0,
    val totalSteps: Int = 0,
    val loss: Float = 0f,
    val bestLoss: Float = Float.MAX_VALUE,
    val tokensPerSec: Double = 0.0,
    val etaSeconds: Long = 0L,
    val ramMb: Long = 0L,
    val batteryPercent: Int = -1,
    val thermalStatus: String = "",
    val deviceTempC: Float? = null,
    val sample: String = "",
    val error: String? = null,
    val lossHistory: List<Float> = emptyList(),
    val elapsedMs: Long = 0L,
    val batchSize: Int = 0,
    val autoPaused: Boolean = false,
    val updatedAt: Long = System.currentTimeMillis()
) {
    val isActive: Boolean
        get() = status == TrainingStatus.PREPARING ||
            status == TrainingStatus.RUNNING ||
            status == TrainingStatus.PAUSED
}

object TrainingStatus {
    const val PREPARING = "preparing"
    const val RUNNING = "running"
    const val PAUSED = "paused"
    const val STOPPED = "stopped"
    const val COMPLETED = "completed"
    const val ERROR = "error"
}
