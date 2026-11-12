package com.modelforge.app.core

import kotlinx.serialization.Serializable

@Serializable
enum class ArchitectureStatus(val label: String) {
    STABLE("Stable"),
    EXPERIMENTAL("Experimental"),
    PLANNED("Planned")
}

@Serializable
enum class Modality(val label: String) {
    TEXT("Text"),
    IMAGE("Image"),
    AUDIO("Audio"),
    VIDEO("Video"),
    DOCUMENT("Documents")
}

@Serializable
enum class DatasetFormat(val label: String) {
    TEXT_CORPUS("Text corpus"),
    IMAGE_FOLDERS("Image folders + labels"),
    IMAGE_CAPTIONS("Images + captions"),
    AUDIO_FOLDERS("Audio folders + labels"),
    AUDIO_TRANSCRIPTS("Audio + transcripts"),
    VIDEO_CAPTIONS("Video + captions"),
    DOCUMENTS("Documents to text"),
    MIXED("Mixed media")
}

@Serializable
data class ArchitectureInfo(
    val id: String,
    val displayName: String,
    val summary: String,
    val status: ArchitectureStatus,
    val inputs: List<Modality>,
    val outputs: List<Modality>,
    val datasetFormat: DatasetFormat,
    val supportedBits: List<Int>,
    val trainable: Boolean = false,
    val unavailableReason: String? = null
) {
    val available: Boolean
        get() = status != ArchitectureStatus.PLANNED && unavailableReason == null
}

data class ArchitectureDerivation(
    val info: ArchitectureInfo,
    val config: ModelConfig?,
    val errorFraction: Double = 0.0,
    val explanation: String = "",
    val warning: String? = null
) {
    val available: Boolean get() = config != null
}

/**
 * One architecture plugin. The traditional transformer is the only path with
 * a native trainer and tests today. NCM and the AI-designed architecture are
 * selectable planning paths; their derive step deliberately returns no native
 * config so the wizard cannot pretend to train them.
 */
interface ArchitecturePlugin {
    val info: ArchitectureInfo

    fun derive(targetParams: Long, maxParams: Long, contextLength: Int? = null): ArchitectureDerivation

    fun defaultTarget(maxParams: Long): Long
}

object ArchitectureRegistry {
    const val GPT_TEXT = "gpt_text"
    const val NCM_TEXT = "ncm_text"
    const val AI_ARCHITECTURE = "ai_architecture"

    private val gpt = object : ArchitecturePlugin {
        override val info = ArchitectureInfo(
            id = GPT_TEXT,
            displayName = "Traditional transformer",
            summary = "Decoder-only transformer for short-form text generation. Stable, tested end to end.",
            status = ArchitectureStatus.STABLE,
            inputs = listOf(Modality.TEXT),
            outputs = listOf(Modality.TEXT),
            datasetFormat = DatasetFormat.TEXT_CORPUS,
            supportedBits = listOf(2, 3, 4, 8, 16, 32),
            trainable = true
        )

        override fun derive(targetParams: Long, maxParams: Long, contextLength: Int?): ArchitectureDerivation {
            val options = ConfigDeriver.Options(maxParams = maxParams)
            val result = ConfigDeriver.derive(
                targetParams,
                if (contextLength != null) {
                    options.copy(contextCandidates = intArrayOf(contextLength))
                } else options
            )
            val config = result.config.copy(architectureId = info.id)
            return ArchitectureDerivation(
                info = info,
                config = config,
                errorFraction = result.errorFraction,
                explanation = result.explanation,
                warning = if (!result.withinTolerance) {
                    "Closest fit is ${String.format("%.1f", result.errorFraction * 100.0)}% from the target."
                } else null
            )
        }

        override fun defaultTarget(maxParams: Long): Long =
            maxParams.coerceIn(1_000_000L, 5_000_000L)
    }

    private fun experimental(
        id: String,
        name: String,
        summary: String,
        inputs: List<Modality>,
        outputs: List<Modality>,
        format: DatasetFormat,
        reason: String
    ) = object : ArchitecturePlugin {
        override val info = ArchitectureInfo(
            id = id,
            displayName = name,
            summary = summary,
            status = ArchitectureStatus.EXPERIMENTAL,
            inputs = inputs,
            outputs = outputs,
            datasetFormat = format,
            supportedBits = emptyList(),
            trainable = false
        )

        override fun derive(targetParams: Long, maxParams: Long, contextLength: Int?) =
            ArchitectureDerivation(
                info = info,
                config = null,
                explanation = reason,
                warning = "Plan only: native training for this architecture is not built yet."
            )

        override fun defaultTarget(maxParams: Long): Long = 0L
    }

    private val plugins: List<ArchitecturePlugin> = listOf(
        gpt,
        experimental(
            NCM_TEXT,
            "NCM",
            "Neural cognitive model with a VQ-VAE and level transformer. The wizard can plan its data, " +
                "but the native NCM trainer and evaluation are not built yet.",
            listOf(Modality.TEXT),
            listOf(Modality.TEXT),
            DatasetFormat.TEXT_CORPUS,
            "Native NCM stages and evaluation are not built yet, so it is a planning path only."
        ),
        experimental(
            AI_ARCHITECTURE,
            "New architecture with AI",
            "AI-designed experimental architecture. The NVIDIA planner can propose its data, rows and " +
                "training recipe; there is no native engine to execute that design on the phone yet.",
            Modality.values().toList(),
            Modality.values().toList(),
            DatasetFormat.MIXED,
            "AI-designed architectures are planning-only until a native engine and tests exist."
        )
    )

    private val byId = plugins.associateBy { it.info.id }

    fun all(): List<ArchitectureInfo> = plugins.map { it.info }

    fun available(): List<ArchitectureInfo> = all().filter { it.available }

    fun plugin(id: String): ArchitecturePlugin = byId[id] ?: gpt

    fun info(id: String): ArchitectureInfo = plugin(id).info

    fun derive(
        id: String,
        targetParams: Long,
        maxParams: Long,
        contextLength: Int? = null
    ): ArchitectureDerivation = plugin(id).derive(targetParams, maxParams, contextLength)

    fun supportedBits(id: String): List<Int> = info(id).supportedBits
}
