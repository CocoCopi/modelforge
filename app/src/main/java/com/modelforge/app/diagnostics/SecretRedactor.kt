package com.modelforge.app.diagnostics

/** Removes credentials from anything that may reach logs or a shared report. */
object SecretRedactor {

    private val authorization = Regex("(?i)(authorization\\s*:\\s*bearer\\s+)[^\\s,;]+")
    private val bearer = Regex("(?i)(bearer\\s+)[A-Za-z0-9._~+/=-]{8,}")
    private val huggingFace = Regex("\\bhf_[A-Za-z0-9]{20,}\\b")
    private val nvidia = Regex("\\bnvapi-[A-Za-z0-9_-]{16,}\\b")
    private val assignment = Regex(
        "(?i)((?:api[_-]?key|access[_-]?token|token|secret|password)\\s*[=:]\\s*)[^\\s,;\"']+"
    )

    fun redact(input: String): String {
        var output = input
        output = authorization.replace(output, "$1[redacted]")
        output = bearer.replace(output, "$1[redacted]")
        output = huggingFace.replace(output, "[redacted-hf-token]")
        output = nvidia.replace(output, "[redacted-nvidia-key]")
        output = assignment.replace(output, "$1[redacted]")
        return output
    }
}
