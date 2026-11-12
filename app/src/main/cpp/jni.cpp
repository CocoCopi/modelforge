#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeVersion(JNIEnv* env, jobject /*thiz*/) {
    return env->NewStringUTF("ModelForge native core scaffold");
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* /*vm*/, void* /*reserved*/) {
    return JNI_VERSION_1_6;
}
