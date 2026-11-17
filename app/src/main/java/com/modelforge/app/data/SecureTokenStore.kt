package com.modelforge.app.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Stores optional Hugging Face and NVIDIA API credentials in
 * EncryptedSharedPreferences. If the keystore is unavailable the store reports
 * that instead of silently writing a plaintext key.
 *
 * Writes return Boolean and commit synchronously: a credential the user
 * believes they saved must not be lost when the process dies immediately
 * afterwards, and a failed write must never be reported as success.
 */
class SecureTokenStore(context: Context) {

    private val prefs: SharedPreferences? = try {
        val masterKey = MasterKey.Builder(context.applicationContext)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context.applicationContext,
            FILE_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    } catch (_: Exception) {
        null
    }

    val isAvailable: Boolean get() = prefs != null

    val token: String?
        get() = prefs?.getString(KEY_TOKEN, null)?.takeIf { it.isNotBlank() }

    val nvidiaToken: String?
        get() = prefs?.getString(KEY_NVIDIA, null)?.takeIf { it.isNotBlank() }

    /** Returns true when the value was persisted (or cleared). */
    fun saveToken(value: String?): Boolean {
        val p = prefs ?: return false
        val editor = p.edit()
        if (value.isNullOrBlank()) editor.remove(KEY_TOKEN) else editor.putString(KEY_TOKEN, value.trim())
        return runCatching { editor.commit() }.getOrDefault(false)
    }

    fun saveNvidiaToken(value: String?): Boolean {
        val p = prefs ?: return false
        val editor = p.edit()
        if (value.isNullOrBlank()) editor.remove(KEY_NVIDIA) else editor.putString(KEY_NVIDIA, value.trim())
        return runCatching { editor.commit() }.getOrDefault(false)
    }

    fun clear(): Boolean = saveToken(null)

    fun clearNvidia(): Boolean = saveNvidiaToken(null)

    companion object {
        private const val FILE_NAME = "modelforge_secrets"
        private const val KEY_TOKEN = "hf_token"
        private const val KEY_NVIDIA = "nvidia_api_key"
    }
}
