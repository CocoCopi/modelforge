package com.modelforge.app.training

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.TrainingStatus

class TrainingRestartReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val store = ModelStore(context)
        val modelId = store.activeModelId() ?: return
        val state = store.loadLiveState(modelId)
        if (state?.status == TrainingStatus.COMPLETED ||
            state?.status == TrainingStatus.STOPPED ||
            state?.status == TrainingStatus.ERROR
        ) {
            store.setActive(modelId, false)
            return
        }
        runCatching { TrainingService.start(context, modelId) }
    }

    companion object {
        fun schedule(context: Context, delayMs: Long) {
            val intent = Intent(context, TrainingRestartReceiver::class.java)
            val pending = PendingIntent.getBroadcast(
                context,
                7,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            val manager = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager
                ?: return
            val at = SystemClock.elapsedRealtime() + delayMs.coerceAtLeast(1_000L)
            runCatching {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    manager.setAndAllowWhileIdle(AlarmManager.ELAPSED_REALTIME_WAKEUP, at, pending)
                } else {
                    manager.set(AlarmManager.ELAPSED_REALTIME_WAKEUP, at, pending)
                }
            }
        }
    }
}
