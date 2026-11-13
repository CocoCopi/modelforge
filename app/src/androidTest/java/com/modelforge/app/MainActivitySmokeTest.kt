package com.modelforge.app

import android.Manifest
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MainActivitySmokeTest {

    @get:Rule(order = 0)
    val notificationPermission: GrantPermissionRule =
        GrantPermissionRule.grant(Manifest.permission.POST_NOTIFICATIONS)

    @get:Rule(order = 1)
    val composeRule = createAndroidComposeRule<MainActivity>()

    @Test
    fun workbenchLaunches() {
        composeRule.onNodeWithText("The workbench").assertIsDisplayed()
        composeRule.onNodeWithText("NEW MODEL").assertIsDisplayed()
    }

    @Test
    fun diagnosticsScreenOpens() {
        composeRule.onNodeWithText("DIAGNOSTICS").performScrollTo().performClick()
        composeRule
            .onNodeWithText("Local evidence for what this build can do. Secrets are redacted.")
            .assertIsDisplayed()
    }
}
