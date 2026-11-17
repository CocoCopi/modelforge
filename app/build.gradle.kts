plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
    id("app.cash.paparazzi")
}

val ciKeystore = System.getenv("MF_KEYSTORE_PATH")
    ?.let { file(it) }
    ?.takeIf { it.exists() }

val keystorePassword = System.getenv("MF_KEYSTORE_PASSWORD")
val keyPassword = System.getenv("MF_KEY_PASSWORD")
val signingMaterialComplete = ciKeystore != null &&
    !keystorePassword.isNullOrBlank() &&
    !keyPassword.isNullOrBlank()

// A release must never be published unsigned. This check lives in Gradle so
// it holds for every CI runner and every task graph, not only for one
// workflow file: if a release task is requested and the signing material is
// missing, the build fails instead of producing an unverifiable APK.
val releaseTaskRequested = gradle.startParameter.taskNames.any {
    it.contains("release", ignoreCase = true)
}
if (releaseTaskRequested && !signingMaterialComplete) {
    throw GradleException(
        "MF_KEYSTORE_PATH / MF_KEYSTORE_PASSWORD / MF_KEY_PASSWORD are required for " +
            "release builds. Refusing to build an unsigned release artifact."
    )
}

android {
    namespace = "com.modelforge.app"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "com.modelforge.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 13
        versionName = "2.2.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-O3", "-ffast-math")
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }

    signingConfigs {
        if (signingMaterialComplete) {
            create("ci") {
                storeFile = ciKeystore
                storePassword = keystorePassword
                keyAlias = System.getenv("MF_KEY_ALIAS") ?: "modelforge"
                keyPassword = keyPassword
            }
        }
    }

    buildTypes {
        debug {
            ndk {
                abiFilters.clear()
                abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
            }
            if (signingMaterialComplete) {
                signingConfig = signingConfigs.getByName("ci")
            }
        }
        release {
            // releaseTaskRequested + missing material already failed above.
            if (signingMaterialComplete) {
                signingConfig = signingConfigs.getByName("ci")
            }
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-opt-in=kotlin.RequiresOptIn")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        resources {
            // Dependency licence and notice files must stay in the APK: some
            // libraries (Apache-2.0, BSD, MIT) require their notices to be
            // distributed with the binary.
            excludes += setOf(
                "/META-INF/versions/9/previous-compilation-data.bin"
            )
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.06.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.2")
    implementation("androidx.lifecycle:lifecycle-service:2.8.2")
    implementation("androidx.navigation:navigation-compose:2.7.7")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter:1.0.0")

    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
    androidTestImplementation(platform("androidx.compose:compose-bom:2024.06.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:rules:1.5.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
