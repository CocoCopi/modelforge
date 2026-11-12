package com.modelforge.app.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * Stores an optional Hugging Face token in EncryptedSharedPreferences.
 * If the keystore is unavailable the store reports that instead of silently
 * writing a plaintext token.
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

    var token: String?
        get() = prefs?.getString(KEY_TOKEN, null)?.takeIf { it.isNotBlank() }
        set(value) {
            val p = prefs ?: return
            p.edit().apply {
                if (value.isNullOrBlank()) remove(KEY_TOKEN) else putString(KEY_TOKEN, value.trim())
            }.apply()
        }

    fun clear() {
        prefs?.edit()?.remove(KEY_TOKEN)?.apply()
    }

    companion object {
        private const val FILE_NAME = "modelforge_secrets"
        private const val KEY_TOKEN = "hf_token"
    }
}
