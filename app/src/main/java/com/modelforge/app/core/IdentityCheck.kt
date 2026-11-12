package com.modelforge.app.core

import com.modelforge.app.training.IdentityDataGenerator

data class IdentityCheckResult(
    val passed: Int,
    val total: Int
) {
    val score: Double get() = if (total <= 0) 0.0 else passed.toDouble() / total.toDouble()
    val needsReinforce: Boolean get() = total > 0 && score < IdentityCheck.PASS_THRESHOLD
}

object IdentityCheck {
    const val QUESTION_COUNT = 20
    const val PASS_THRESHOLD = 0.80

    fun questions(languages: List<String>): List<String> =
        IdentityDataGenerator.checkQuestions(languages, QUESTION_COUNT)

    fun isIdentityAnswer(name: String, creator: String, response: String): Boolean {
        val text = response.lowercase()
        val cleanName = name.trim().lowercase()
        val cleanCreator = creator.trim().lowercase()
        if (cleanName.isEmpty() || cleanCreator.isEmpty()) return false
        return text.contains(cleanName) && text.contains(cleanCreator)
    }

    fun evaluate(
        name: String,
        creator: String,
        responses: List<String>
    ): IdentityCheckResult {
        val passed = responses.count { isIdentityAnswer(name, creator, it) }
        return IdentityCheckResult(passed = passed, total = responses.size)
    }
}
