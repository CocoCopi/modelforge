package com.modelforge.app.training

import com.modelforge.app.data.TrainingLiveState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** In-process live training updates for the UI. The service also persists state. */
object TrainingBus {
    private val _state = MutableStateFlow<TrainingLiveState?>(null)
    val state: StateFlow<TrainingLiveState?> = _state.asStateFlow()

    fun publish(state: TrainingLiveState) {
        _state.value = state
    }

    fun peek(): TrainingLiveState? = _state.value
}
