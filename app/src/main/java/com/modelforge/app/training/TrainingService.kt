package com.modelforge.app.training

import android.app.Service
import android.content.Intent
import android.os.IBinder

/**
 * Placeholder for the Phase 4 foreground training service. The real service starts
 * training, holds a partial wake lock, posts progress and survives screen-off.
 */
class TrainingService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        stopSelf(startId)
        return START_NOT_STICKY
    }
}
