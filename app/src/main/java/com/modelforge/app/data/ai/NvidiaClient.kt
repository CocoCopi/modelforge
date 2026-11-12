package com.modelforge.app.data.ai

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

data class AiMessage(val role: String, val content: String)

class AiException(message: String, cause: Throwable? = null) : Exception(message, cause)

object NvidiaStreamParser {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    fun delta(data: String): String? {
        val root = runCatching { json.parseToJsonElement(data) as? JsonObject }.getOrNull() ?: return null
        val choices = root["choices"] as? JsonArray ?: return null
        val delta = (choices.firstOrNull() as? JsonObject)?.get("delta") as? JsonObject ?: return null
        val content = delta["content"] ?: return null
        return when (content) {
            is JsonPrimitive -> content.contentOrNull ?: content.toString().trim('"')
            is JsonArray -> content.mapNotNull { part ->
                val obj = part as? JsonObject ?: return@mapNotNull null
                (obj["text"] as? JsonPrimitive)?.contentOrNull
            }.joinToString("")
            else -> null
        }?.takeIf { it.isNotEmpty() }
    }
}

class NvidiaClient(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
) {
    suspend fun streamChat(
        apiKey: String,
        messages: List<AiMessage>,
        onDelta: (String) -> Unit = {}
    ): String = withContext(Dispatchers.IO) {
        if (apiKey.isBlank()) throw AiException("No NVIDIA API key saved.")
        val payload = buildJsonObject {
            put("model", MODEL)
            put("max_tokens", 16384)
            put("seed", 0)
            put("stream", true)
            put("temperature", 1.0)
            put("top_p", 0.95)
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
        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    val detail = response.body?.string().orEmpty().take(400)
                    throw AiException("NVIDIA returned HTTP ${response.code}. ${detail.trim()}")
                }
                val body = response.body ?: throw AiException("NVIDIA returned an empty response.")
                val source = body.source()
                val out = StringBuilder()
                while (true) {
                    val line = source.readUtf8Line() ?: break
                    if (!line.startsWith("data:")) continue
                    val data = line.removePrefix("data:").trim()
                    if (data.isBlank()) continue
                    if (data == "[DONE]") break
                    val delta = NvidiaStreamParser.delta(data) ?: continue
                    out.append(delta)
                    onDelta(delta)
                }
                if (out.isBlank()) throw AiException("The AI returned no text.")
                out.toString()
            }
        } catch (t: AiException) {
            throw t
        } catch (t: Throwable) {
            throw AiException("Could not reach NVIDIA: ${t.message ?: t.javaClass.simpleName}", t)
        }
    }

    companion object {
        const val INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
        const val MODEL = "minimaxai/minimax-m3.1-flash"
    }
}
