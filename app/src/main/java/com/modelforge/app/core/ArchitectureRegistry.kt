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
    VIDEO("Video")
}

@Serializable
enum class DatasetFormat(val label: String) {
    TEXT_CORPUS("Text corpus"),
    IMAGE_FOLDERS("Image folders + labels"),
    IMAGE_CAPTIONS("Images + captions"),
    AUDIO_FOLDERS("Audio folders + labels"),
    AUDIO_TRANSCRIPTS("Audio + transcripts"),
    VIDEO_CAPTIONS("Video + captions")
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
 * One architecture plugin. Text GPT is the only implementation today; the
 * planned entries are visible in the wizard so the roadmap is honest, but they
 * cannot be selected until their native plugin and tests exist.
 */
interface ArchitecturePlugin {
    val info: ArchitectureInfo

    fun derive(targetParams: Long, maxParams: Long, contextLength: Int? = null): ArchitectureDerivation

    fun defaultTarget(maxParams: Long): Long
}

object ArchitectureRegistry {
    const val GPT_TEXT = "gpt_text"
    const val NCM_TEXT = "ncm_text"
    const val IMAGE_CLASSIFIER = "image_classifier"
    const val AUDIO_CLASSIFIER = "audio_classifier"
    const val CTC_SPEECH = "ctc_speech"
    const val IMAGE_CAPTIONING = "image_captioning"
    const val TEXT_TO_IMAGE = "text_to_image"
    const val TEXT_TO_SPEECH = "text_to_speech"
    const val VIDEO = "video"

    private val gpt = object : ArchitecturePlugin {
        override val info = ArchitectureInfo(
            id = GPT_TEXT,
            displayName = "GPT for text",
            summary = "Decoder-only transformer for short-form text generation. Stable, tested end to end.",
            status = ArchitectureStatus.STABLE,
            inputs = listOf(Modality.TEXT),
            outputs = listOf(Modality.TEXT),
            datasetFormat = DatasetFormat.TEXT_CORPUS,
            supportedBits = listOf(2, 3, 4, 8, 16, 32)
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

    private fun planned(
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
            status = ArchitectureStatus.PLANNED,
            inputs = inputs,
            outputs = outputs,
            datasetFormat = format,
            supportedBits = emptyList(),
            unavailableReason = reason
        )

        override fun derive(targetParams: Long, maxParams: Long, contextLength: Int?) =
            ArchitectureDerivation(info = info, config = null, explanation = reason)

        override fun defaultTarget(maxParams: Long): Long = 0L
    }

    private val plugins: List<ArchitecturePlugin> = listOf(
        gpt,
        planned(
            NCM_TEXT,
            "NCM for text",
            "Task-specific neural cognitive model with a VQ-VAE and level transformer.",
            listOf(Modality.TEXT),
            listOf(Modality.TEXT),
            DatasetFormat.TEXT_CORPUS,
            "Native NCM stages and evaluation are not built yet."
        ),
        planned(
            IMAGE_CLASSIFIER,
            "Image classifier",
            "Small convolutional classifier for folders of labelled images.",
            listOf(Modality.IMAGE),
            listOf(Modality.TEXT),
            DatasetFormat.IMAGE_FOLDERS,
            "Image preprocessing and the classifier engine are not built yet."
        ),
        planned(
            AUDIO_CLASSIFIER,
            "Audio classifier",
            "Keyword spotting and short-clip classification.",
            listOf(Modality.AUDIO),
            listOf(Modality.TEXT),
            DatasetFormat.AUDIO_FOLDERS,
            "Audio decoding and the classifier engine are not built yet."
        ),
        planned(
            CTC_SPEECH,
            "Speech to text",
            "Character-level CTC recogniser for short recordings.",
            listOf(Modality.AUDIO),
            listOf(Modality.TEXT),
            DatasetFormat.AUDIO_TRANSCRIPTS,
            "The CTC engine is not built yet."
        ),
        planned(
            IMAGE_CAPTIONING,
            "Image captioning",
            "Vision encoder plus a small text decoder.",
            listOf(Modality.IMAGE),
            listOf(Modality.TEXT),
            DatasetFormat.IMAGE_CAPTIONS,
            "The vision encoder is not built yet."
        ),
        planned(
            TEXT_TO_IMAGE,
            "Text to image",
            "A very small diffusion model for low-resolution images.",
            listOf(Modality.TEXT),
            listOf(Modality.IMAGE),
            DatasetFormat.IMAGE_CAPTIONS,
            "Planned after the classifier and captioning paths."
        ),
        planned(
            TEXT_TO_SPEECH,
            "Text to speech",
            "Tiny phoneme or character acoustic model.",
            listOf(Modality.TEXT),
            listOf(Modality.AUDIO),
            DatasetFormat.AUDIO_TRANSCRIPTS,
            "Planned after the speech recognition path."
        ),
        planned(
            VIDEO,
            "Video",
            "Frame sampling plus a small temporal model.",
            listOf(Modality.VIDEO),
            listOf(Modality.TEXT),
            DatasetFormat.VIDEO_CAPTIONS,
            "Planned last; video memory and decoding come first."
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
