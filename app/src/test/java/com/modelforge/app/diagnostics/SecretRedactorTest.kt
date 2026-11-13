package com.modelforge.app.diagnostics

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SecretRedactorTest {

    @Test
    fun redactsHuggingFaceTokens() {
        val key = "hf_abcdefghijklmnopqrstuvwxyz012345"
        val output = SecretRedactor.redact("token=$key failed")
        assertFalse(output.contains(key))
        assertTrue(output.contains("[redacted"))
    }

    @Test
    fun redactsNvidiaKeysAndBearerHeaders() {
        val key = "nvapi-abcdefghijklmnopqrstuvwxyz012345"
        val output = SecretRedactor.redact(
            "Authorization: Bearer abcdefghijklmnop\napi_key: $key"
        )
        assertFalse(output.contains("abcdefghijklmnop"))
        assertFalse(output.contains(key))
        assertTrue(output.contains("[redacted]"))
    }

    @Test
    fun leavesDiagnosticTextIntact() {
        val text = "Hugging Face token saved=true; secure storage unavailable"
        assertTrue(SecretRedactor.redact(text).contains("Hugging Face token saved=true"))
    }
}
