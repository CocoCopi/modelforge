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
import kotlinx.coroutines.delay

/**
 * In-process service updates arrive on every step via TrainingBus. Disk state is
 * the fallback after a process restart or when the service has not published yet.
 */
@Composable
fun rememberTrainingLiveState(vm: ModelForgeViewModel, modelId: String): TrainingLiveState? {
    val bus by TrainingBus.state.collectAsState()
    var stored by remember(modelId) { mutableStateOf(vm.liveState(modelId)) }
    LaunchedEffect(modelId) {
        while (true) {
            stored = vm.liveState(modelId)
            delay(1_000)
        }
    }
    return bus?.takeIf { it.modelId == modelId } ?: stored
}
