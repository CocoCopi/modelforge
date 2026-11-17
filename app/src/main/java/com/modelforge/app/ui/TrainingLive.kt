package com.modelforge.app.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.training.TrainingBus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * In-process service updates arrive on every step via TrainingBus. Disk state is
 * the fallback after a process restart or when the service has not published yet.
 * Reading it happens on Dispatchers.IO so composition never blocks on JSON I/O.
 */
@Composable
fun rememberTrainingLiveState(vm: ModelForgeViewModel, modelId: String): TrainingLiveState? {
    val bus by TrainingBus.state.collectAsState()
    var stored by remember(modelId) { mutableStateOf<TrainingLiveState?>(null) }
    LaunchedEffect(modelId) {
        while (true) {
            stored = withContext(Dispatchers.IO) { vm.liveState(modelId) }
            delay(1_000)
        }
    }
    return bus?.takeIf { it.modelId == modelId } ?: stored
}
