package com.modelforge.app.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ModelForgeRoot(vm: ModelForgeViewModel = viewModel()) {
    val screen by vm.screen.collectAsState()

    BackHandler(enabled = screen !is Screen.Home) {
        if (screen is Screen.Wizard) {
            vm.wizardBack()
        } else {
            vm.goHome()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        when (val current = screen) {
            is Screen.Home -> HomeScreen(vm)
            is Screen.Wizard -> WizardScreen(vm)
            is Screen.Training -> TrainingScreen(vm, current.modelId)
            is Screen.ModelDetail -> ModelDetailScreen(vm, current.modelId)
        }
    }
}
