package com.modelforge.app.data.ai

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

data class AiMessage(val role: String, val content: String)

class AiException(message: String, cause: Throwable? = null) : Exception(message, cause)

interface AiChat {
    suspend fun streamChat(
        apiKey: String,
        messages: List<AiMessage>,
        onDelta: (String) -> Unit = {},
        onActivity: () -> Unit = {}
    ): String
}

object NvidiaStreamParser {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    private fun firstChoice(data: String): JsonObject? {
        val root = runCatching { json.parseToJsonElement(data.trim()) as? JsonObject }.getOrNull() ?: return null
        val choices = root["choices"] as? JsonArray ?: return null
        return choices.firstOrNull() as? JsonObject
    }

    private fun textOf(element: kotlinx.serialization.json.JsonElement?): String? = when (element) {
        is JsonPrimitive -> element.contentOrNull ?: element.toString().trim('"')
        is JsonArray -> element.mapNotNull { part ->
            val obj = part as? JsonObject ?: return@mapNotNull null
            (obj["text"] as? JsonPrimitive)?.contentOrNull
        }.joinToString("")
        else -> null
    }?.takeIf { it.isNotEmpty() }

    fun delta(data: String): String? {
        val delta = firstChoice(data)?.get("delta") as? JsonObject ?: return null
        return textOf(delta["content"])
    }

    fun reasoningDelta(data: String): String? {
        val delta = firstChoice(data)?.get("delta") as? JsonObject ?: return null
        return textOf(delta["reasoning_content"] ?: delta["reasoning"])
    }

    fun fullContent(data: String): String? {
        val choice = firstChoice(data) ?: return null
        val message = choice["message"] as? JsonObject
        return textOf(message?.get("content")) ?: textOf(choice["text"])
    }
}

internal object NvidiaRetryPolicy {
    const val MAX_ATTEMPTS = 4

    fun isRetryable(code: Int): Boolean = code == 429 || code in 500..599

    fun waitSeconds(code: Int, retryAfter: Long?, attempt: Int): Long {
        val fallback = if (code == 429) {
            longArrayOf(20L, 45L, 60L)
        } else {
            longArrayOf(5L, 15L, 30L)
        }
        return retryAfter?.coerceIn(1L, 120L)
            ?: fallback[attempt.coerceIn(0, fallback.lastIndex)]
    }
}

private class NvidiaRetryableException(
    val code: Int,
    val retryAfter: Long?,
    message: String
) : Exception(message)

class NvidiaClient(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .callTimeout(300, TimeUnit.SECONDS)
        .build(),
    private val onRetry: (String) -> Unit = {}
) : AiChat {
    @Volatile
    private var activeCall: Call? = null

    fun cancel() {
        activeCall?.cancel()
    }

    override suspend fun streamChat(
        apiKey: String,
        messages: List<AiMessage>,
        onDelta: (String) -> Unit,
        onActivity: () -> Unit
    ): String = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) throw AiException("No NVIDIA API key saved.")
        val payload = buildJsonObject {
            put("model", MODEL)
            put("max_tokens", 4096)
            put("seed", 0)
            put("stream", true)
            put("temperature", 0.4)
            put("top_p", 0.95)
            put("reasoning_effort", "none")
            put("messages", buildJsonArray {
                messages.forEach { message ->
                    add(buildJsonObject {
                        put("role", message.role)
                        put("content", message.content)
                    })
                }
            })
        }
        val request = Request.Builder()
            .url(INVOKE_URL)
            .header("Authorization", "Bearer ${apiKey.trim()}")
            .header("Accept", "text/event-stream")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        var attempt = 0
        while (true) {
            val call = client.newCall(request)
            activeCall = call
            val failure = try {
                call.execute().use { response ->
                    if (!response.isSuccessful) {
                        val code = response.code
                        val detail = response.body?.string().orEmpty().take(500).trim()
                        val hint = when (code) {
                            401, 403 -> "Check that the NVIDIA API key is valid and has access to $MODEL."
                            404 -> "The model $MODEL may not be available to this account."
                            429 -> "NVIDIA rate limit reached. Wait a minute and retry."
                            in 500..599 -> "NVIDIA is temporarily unavailable. The app retried " +
                                "automatically; wait a minute and try again."
                            else -> "The hosted request failed."
                        }
                        val message = "NVIDIA HTTP $code: $detail $hint"
                        if (NvidiaRetryPolicy.isRetryable(code)) {
                            throw NvidiaRetryableException(
                                code,
                                response.header("Retry-After")?.toLongOrNull(),
                                message
                            )
                        }
                        throw AiException(message)
                    }
                    val body = response.body ?: throw AiException("NVIDIA returned an empty response.")
                    val contentType = response.header("Content-Type").orEmpty()
                    if (!contentType.contains("text/event-stream", ignoreCase = true)) {
                        val raw = body.string()
                        val content = NvidiaStreamParser.fullContent(raw)
                            ?: throw AiException("NVIDIA returned an unexpected response: ${raw.take(300)}")
                        onActivity()
                        onDelta(content)
                        return@withContext content
                    }
                    val source = body.source()
                    val out = StringBuilder()
                    while (true) {
                        val line = source.readUtf8Line() ?: break
                        if (!line.startsWith("data:")) continue
                        val data = line.removePrefix("data:").trim()
                        if (data.isBlank()) continue
                        if (data == "[DONE]") break
                        onActivity()
                        val delta = NvidiaStreamParser.delta(data)
                        if (delta != null) {
                            out.append(delta)
                            onDelta(delta)
                        }
                    }
                    if (out.isBlank()) {
                        throw AiException(
                            "NVIDIA finished without returning plan text. The model may have spent its budget " +
                                "on hidden reasoning. Retry, or try again with a shorter model plan."
                        )
                    }
                    return@withContext out.toString()
                }
                null
            } catch (t: NvidiaRetryableException) {
                t
            } catch (t: AiException) {
                throw t
            } catch (t: java.net.SocketTimeoutException) {
                throw AiException(
                    "NVIDIA timed out. If the progress counter stayed at 0 events, this phone could not " +
                        "reach integrate.api.nvidia.com; try switching Wi-Fi/mobile data or VPN. If events " +
                        "were arriving, the model took too long.",
                    t
                )
            } catch (t: java.net.UnknownHostException) {
                throw AiException(
                    "Could not resolve integrate.api.nvidia.com. Check the connection, DNS or VPN.",
                    t
                )
            } catch (t: java.net.ConnectException) {
                throw AiException(
                    "Could not connect to integrate.api.nvidia.com. Check the connection or VPN.",
                    t
                )
            } catch (t: javax.net.ssl.SSLException) {
                throw AiException(
                    "Secure connection to NVIDIA failed. Check the phone's date/time and VPN.",
                    t
                )
            } catch (t: Throwable) {
                val message = t.message ?: t.javaClass.simpleName
                throw AiException(
                    if (message.contains("Canceled", ignoreCase = true) ||
                        message.contains("Socket closed", ignoreCase = true)
                    ) "AI request was cancelled."
                    else "Could not reach NVIDIA: $message",
                    t
                )
            } finally {
                if (activeCall === call) activeCall = null
            }

            if (failure == null) {
                throw AiException("NVIDIA request ended unexpectedly. Retry.")
            }
            attempt++
            if (attempt >= NvidiaRetryPolicy.MAX_ATTEMPTS) {
                throw AiException(failure.message ?: "NVIDIA is temporarily unavailable.")
            }
            val seconds = NvidiaRetryPolicy.waitSeconds(
                failure.code,
                failure.retryAfter,
                attempt - 1
            )
            onRetry("NVIDIA is temporarily unavailable; retrying in ${seconds}s")
            delay(seconds * 1_000L)
        }
        @Suppress("UNREACHABLE_CODE")
        throw AiException("NVIDIA request ended unexpectedly. Retry.")
    }

    companion object {
        const val INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
        const val MODEL = "minimaxai/minimax-m3.1-flash"
    }
}
