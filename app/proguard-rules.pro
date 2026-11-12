# Keep JNI entry points; native code resolves these names at runtime.
-keep class com.modelforge.app.nativebridge.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep the serializable model, dataset, preset and AI plan shapes. These are
# persisted as JSON and some are also used by Retrofit and Kotlin serialization.
-keep class com.modelforge.app.core.** { *; }
-keep class com.modelforge.app.data.** { *; }

# Keep manifest-referenced components and the training service callbacks.
-keep class com.modelforge.app.ModelForgeApp { *; }
-keep class com.modelforge.app.MainActivity { *; }
-keep class com.modelforge.app.training.TrainingService { *; }
-keep class com.modelforge.app.training.BootReceiver { *; }
-keep class com.modelforge.app.training.TrainingRestartReceiver { *; }

# Retrofit and kotlinx.serialization rely on runtime annotations and signatures.
-keepattributes Signature, InnerClasses, EnclosingMethod, *Annotation*
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Retrofit interfaces are reached through reflection.
-keep interface com.modelforge.app.data.hf.HfApi { *; }
-keepclassmembers class * {
    @kotlinx.serialization.SerialName <fields>;
}
-keepclasseswithmembers class * {
    kotlinx.serialization.KSerializer serializer(...);
}

# OkHttp optional platform integrations are not used on Android.
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
-dontwarn okhttp3.internal.platform.**
