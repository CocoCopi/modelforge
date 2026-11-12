package com.modelforge.app.training

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.TrainingStatus

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED &&
            intent.action != Intent.ACTION_MY_PACKAGE_REPLACED
        ) {
            return
        }
        val store = ModelStore(context)
        val modelId = store.activeModelId() ?: return
        val manifest = store.loadManifest(modelId) ?: return
        if (!manifest.settings.autoResume) {
            store.setActive(modelId, false)
            return
        }
        val state = store.loadLiveState(modelId)
        if (state != null && !state.isActive) {
            store.setActive(modelId, false)
            return
        }
        if (state?.status == TrainingStatus.COMPLETED ||
            state?.status == TrainingStatus.STOPPED ||
            state?.status == TrainingStatus.ERROR
        ) {
            store.setActive(modelId, false)
            return
        }
        runCatching { TrainingService.start(context, modelId) }
    }
}
