package com.modelforge.app.ui

enum class WizardStep(val label: String, val question: String, val help: String) {
    IDENTITY(
        "Identity",
        "Who is this model?",
        "A name and a creator make the model honest about where it came from."
    ),
    SIZE(
        "Size",
        "How large should it be?",
        "Drag the ruler. Larger models learn more but take longer and use more memory."
    ),
    ARCHITECTURE(
        "Architecture",
        "What kind of model is it?",
        "Only tested paths can be selected. Planned paths show why they are disabled."
    ),
    MODALITIES(
        "Modalities",
        "What goes in, and what comes out?",
        "Modalities come from the architecture, so they cannot drift out of sync."
    ),
    LANGUAGES(
        "Languages",
        "Which languages will it practise?",
        "Choose the languages present in your data. More languages need more data."
    ),
    CONTEXT(
        "Context",
        "How much text should it hold?",
        "Context is the tokens the model can see at once. Longer context costs memory."
    ),
    SPEED(
        "Response speed",
        "How should it answer?",
        "Speed changes inference only. Training quality is unaffected."
    ),
    REASONING(
        "Reasoning",
        "Should it think before answering?",
        "Reasoning reserves room for a thinking block. Very small models often gain little."
    ),
    TOOLS(
        "Tool calling",
        "Should it ask for tools?",
        "Tool calls are trained examples; the app never runs a tool by itself."
    ),
    PERSONA(
        "Persona",
        "How should it behave?",
        "A short behaviour note becomes part of the identity and is included in training."
    ),
    COMPRESSION(
        "Compression",
        "How small should the file be?",
        "Quantization shrinks linear weights. The bit density shows the tradeoff."
    ),
    DATASETS(
        "Datasets",
        "What should it learn from?",
        "Search Hugging Face, inspect the columns, and add one or more datasets."
    ),
    BUDGET(
        "Training budget",
        "How long should training run?",
        "More steps mean more practice. The estimate uses this model's real disk layout."
    ),
    REVIEW(
        "Review",
        "Ready to train?",
        "Check the decisions. You can still go back before training starts."
    );

    companion object {
        val entries = values().toList()
        val lastIndex = entries.lastIndex
        fun at(index: Int): WizardStep = entries[index.coerceIn(0, lastIndex)]
    }
}
