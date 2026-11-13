package com.modelforge.app

import android.app.Application
import android.os.Build
import com.modelforge.app.diagnostics.CrashGuard
import com.modelforge.app.diagnostics.ForgeLog

class ModelForgeApp : Application() {

    override fun onCreate() {
        super.onCreate()
        CrashGuard.install(this)
        ForgeLog.info(
            "app",
            "ModelForge ${BuildConfig.VERSION_NAME} started on ${Build.MANUFACTURER} ${Build.MODEL}, " +
                "SDK ${Build.VERSION.SDK_INT}, ${Build.SUPPORTED_ABIS.joinToString("/")}"
        )
    }
}
