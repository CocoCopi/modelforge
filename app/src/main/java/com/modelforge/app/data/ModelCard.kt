package com.modelforge.app.data

import com.modelforge.app.core.ArchitectureRegistry
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ModelCard {
    fun build(
        manifest: ModelManifest,
        state: TrainingLiveState? = null,
        generatedAt: Long = System.currentTimeMillis()
    ): String {
        val config = manifest.config
        val info = ArchitectureRegistry.info(manifest.config.architectureId)
        val date = SimpleDateFormat("yyyy-MM-dd HH:mm 'UTC'", Locale.US)
            .format(Date(generatedAt))
        val settings = manifest.settings
        val datasets = if (manifest.datasets.isEmpty()) {
            "No datasets recorded."
        } else {
            manifest.datasets.joinToString("\n") { entry ->
                "- `${entry.datasetId}` (${entry.config}/${entry.split}), " +
                    "${entry.rows} rows, weight ${String.format(Locale.US, "%.1f", entry.weight)}x, " +
                    "licence ${entry.license ?: "unknown"}" +
                    if (entry.nonCommercial) ", non-commercial only" else ""
            }
        }
        val behaviour = buildList {
            add("- Response speed: ${manifest.behavior.speedMode}")
            add(
                "- Reasoning: " + if (manifest.behavior.reasoningEnabled) {
                    "${manifest.behavior.reasoningRoom} reserved tokens"
                } else {
                    "off"
                }
            )
            add("- Tool calling: " + if (manifest.behavior.toolCalling) "trained examples" else "off")
            if (manifest.behavior.persona.isNotBlank()) {
                add("- Persona: ${manifest.behavior.persona.trim()}")
            }
        }.joinToString("\n")
        val training = buildString {
            append("- Steps: ${settings.steps}\n")
            append("- Batch size: ${settings.batchSize}\n")
            append("- Sequence length: ${settings.seqLength}\n")
            append("- Learning rate: ${settings.learningRate}\n")
            append("- Identity fraction: ${String.format(Locale.US, "%.1f", settings.identityFraction * 100)}%\n")
            if (state != null) {
                append("- Last status: ${state.status}\n")
                if (state.step > 0) append("- Last step: ${state.step} / ${state.totalSteps}\n")
                if (state.bestLoss != Float.MAX_VALUE) {
                    append("- Best loss: ${String.format(Locale.US, "%.6f", state.bestLoss)}\n")
                }
            }
        }.trimEnd()
        return """
# ${manifest.name}

Created by ${manifest.creator.ifBlank { "an unnamed creator" }}.

- Architecture: ${info.displayName} (${info.status.label})
- Parameters: ${config.paramCount()}
- Layers: ${config.layers}
- Width: ${config.dModel}, ${config.heads} heads of ${config.headDim}
- Feed-forward: ${config.dFf}
- Context: ${config.contextLength} tokens
- Vocabulary: ${config.vocabSize}
- Export precision: ${config.bits}-bit${if (config.qat) " with QAT" else ""}
- Languages: ${manifest.languages.joinToString(", ").ifBlank { "not recorded" }}
- Created: ${SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date(manifest.createdAt))}
- Card generated: $date

## Intended use

A small, from-scratch language model trained on a phone. It is intended for
learning, experimentation and short local text generation. It is not a general
assistant, not benchmarked against production models, and should not be used as
the only source for decisions that affect people.

## Training

$training

## Behaviour

$behaviour

## Datasets and licences

$datasets

You are responsible for checking and following each dataset's licence,
including non-commercial or attribution terms, before sharing this model or its
outputs. A dataset licence can restrict the model weights as well as the data.

## Limitations

- Trained from scratch on a small device budget; repetition and shaky grammar
  are expected.
- Samples are produced during training, but no broad quality benchmark has been
  run.
- The model can be confidently wrong. Verify important answers.
- Identity is reinforced with synthetic conversations; a low identity check
  score is reported rather than hidden.
""".trimStart()
    }
}
