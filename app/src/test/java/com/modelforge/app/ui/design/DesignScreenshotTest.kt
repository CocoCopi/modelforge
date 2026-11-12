package com.modelforge.app.ui.design

import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.modelforge.app.ui.theme.ModelForgeTheme
import java.io.File
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test

class DesignScreenshotTest {
    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = DeviceConfig.PIXEL_5,
        theme = "android:Theme.Material.Light.NoActionBar"
    )

    private val golden = File(
        "src/test/snapshots/images/" +
            "com.modelforge.app.ui.design_DesignScreenshotTest_design_gallery.png"
    )

    @Test
    fun design_gallery() {
        assumeTrue(
            "Run recordPaparazziDebug to create the first golden image.",
            System.getenv("PAPARAZZI_RECORD") == "true" || golden.exists()
        )
        paparazzi.snapshot(name = "design_gallery") {
            ModelForgeTheme {
                DesignGalleryScreen()
            }
        }
    }
}
