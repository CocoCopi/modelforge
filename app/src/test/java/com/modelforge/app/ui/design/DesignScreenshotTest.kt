package com.modelforge.app.ui.design

import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.modelforge.app.ui.theme.ModelForgeTheme
import org.junit.Rule
import org.junit.Test

class DesignScreenshotTest {
    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = DeviceConfig.PIXEL_5,
        theme = "android:Theme.Material.Light.NoActionBar"
    )

    @Test
    fun design_gallery() {
        paparazzi.snapshot(name = "design_gallery") {
            ModelForgeTheme {
                DesignGalleryScreen()
            }
        }
    }
}
