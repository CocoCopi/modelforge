#include "core/gpt.h"

#include <jni.h>

#include <cstdint>
#include <new>
#include <string>
#include <vector>

using mf::GPT;
using mf::ModelConfig;

namespace {

GPT* from_handle(jlong handle) {
    return reinterpret_cast<GPT*>(static_cast<intptr_t>(handle));
}

std::string to_string(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string out = chars ? chars : "";
    if (chars) env->ReleaseStringUTFChars(s, chars);
    return out;
}

bool read_ints(JNIEnv* env, jintArray arr, std::vector<int>& out) {
    if (arr == nullptr) return false;
    const jsize n = env->GetArrayLength(arr);
    out.resize(static_cast<size_t>(n));
    env->GetIntArrayRegion(arr, 0, n, reinterpret_cast<jint*>(out.data()));
    return true;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeVersion(JNIEnv* env, jobject /*thiz*/) {
    return env->NewStringUTF("ModelForge C++17 engine (GPT, AdamW, QAT-ready)");
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeCreate(
    JNIEnv* /*env*/, jobject /*thiz*/,
    jint vocab, jint context, jint d_model, jint layers, jint heads, jint d_ff,
    jboolean tied, jlong seed) {
    ModelConfig cfg;
    cfg.vocab = vocab;
    cfg.context = context;
    cfg.d_model = d_model;
    cfg.layers = layers;
    cfg.heads = heads;
    cfg.d_ff = d_ff;
    cfg.tied_embeddings = tied != JNI_FALSE;
    if (!cfg.valid()) return 0;
    try {
        auto* model = new (std::nothrow) GPT();
        if (!model) return 0;
        if (!model->init(cfg, static_cast<uint64_t>(seed))) {
            delete model;
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<intptr_t>(model));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeFree(JNIEnv* /*env*/, jobject /*thiz*/,
                                                             jlong handle) {
    delete from_handle(handle);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeConfig(JNIEnv* env, jobject /*thiz*/,
                                                               jlong handle) {
    GPT* m = from_handle(handle);
    if (!m) return nullptr;
    const ModelConfig& c = m->config();
    jint vals[9] = {c.vocab, c.context, c.d_model, c.layers, c.heads, c.d_ff,
                    c.tied_embeddings ? 1 : 0, m->qat_bits(), m->qat_group()};
    jintArray result = env->NewIntArray(9);
    if (result) env->SetIntArrayRegion(result, 0, 9, vals);
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeParamCount(JNIEnv* /*env*/, jobject /*thiz*/,
                                                                   jlong handle) {
    GPT* m = from_handle(handle);
    return m ? static_cast<jlong>(m->param_count()) : 0;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeTrainStep(
    JNIEnv* env, jobject /*thiz*/, jlong handle,
    jintArray tokens, jintArray targets, jint B, jint T, jfloat lr, jfloat weight_decay) {
    GPT* m = from_handle(handle);
    if (!m || B <= 0 || T <= 0) return -1.0f;
    try {
        std::vector<int> tok, tgt;
        if (!read_ints(env, tokens, tok) || !read_ints(env, targets, tgt)) return -1.0f;
        const size_t need = static_cast<size_t>(B) * static_cast<size_t>(T);
        if (tok.size() < need || tgt.size() < need) return -1.0f;
        return m->train_step(tok.data(), tgt.data(), B, T, lr, weight_decay);
    } catch (...) {
        return -1.0f;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeStep(JNIEnv* /*env*/, jobject /*thiz*/,
                                                             jlong handle) {
    GPT* m = from_handle(handle);
    return m ? static_cast<jlong>(m->step()) : 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeSetQAT(JNIEnv* /*env*/, jobject /*thiz*/,
                                                               jlong handle, jint bits, jint group) {
    GPT* m = from_handle(handle);
    if (m) m->set_qat(bits, group);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeGenerate(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jintArray prompt,
    jint max_new, jfloat temperature, jint top_k, jfloat top_p, jlong seed) {
    GPT* m = from_handle(handle);
    if (!m) return nullptr;
    try {
        std::vector<int> p;
        if (!read_ints(env, prompt, p)) return nullptr;
        for (int& v : p) {
            if (v < 0) v = 0;
            if (v >= m->config().vocab) v = m->config().vocab - 1;
        }
        const auto out = m->generate(p, max_new, temperature, top_k, top_p,
                                     static_cast<uint64_t>(seed));
        jintArray result = env->NewIntArray(static_cast<jsize>(out.size()));
        if (result && !out.empty()) {
            env->SetIntArrayRegion(result, 0, static_cast<jsize>(out.size()),
                                   reinterpret_cast<const jint*>(out.data()));
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeSaveCheckpoint(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jstring path, jboolean with_optimizer) {
    GPT* m = from_handle(handle);
    if (!m) return JNI_FALSE;
    try {
        const std::string p = to_string(env, path);
        if (p.empty()) return JNI_FALSE;
        return m->save_checkpoint(p, with_optimizer != JNI_FALSE) ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeLoadCheckpoint(
    JNIEnv* env, jobject /*thiz*/, jstring path, jboolean with_optimizer) {
    try {
        const std::string p = to_string(env, path);
        if (p.empty()) return 0;
        auto* model = new (std::nothrow) GPT();
        if (!model) return 0;
        if (!model->load_checkpoint(p, with_optimizer != JNI_FALSE)) {
            delete model;
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<intptr_t>(model));
    } catch (...) {
        return 0;
    }
}


extern "C" JNIEXPORT jboolean JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeSavePacked(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jstring path, jint bits, jint group) {
    GPT* m = from_handle(handle);
    if (!m) return JNI_FALSE;
    try {
        const std::string p = to_string(env, path);
        if (p.empty()) return JNI_FALSE;
        return m->save_packed(p, bits, group) ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativeLoadPacked(
    JNIEnv* env, jobject /*thiz*/, jstring path) {
    try {
        const std::string p = to_string(env, path);
        if (p.empty()) return 0;
        auto* model = new (std::nothrow) PackedModel();
        if (!model) return 0;
        if (!model->load(p)) {
            delete model;
            return 0;
        }
        return static_cast<jlong>(reinterpret_cast<intptr_t>(model));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativePackedFree(JNIEnv* /*env*/, jobject /*thiz*/,
                                                                  jlong handle) {
    delete reinterpret_cast<PackedModel*>(static_cast<intptr_t>(handle));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativePackedParamCount(
    JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    auto* m = reinterpret_cast<PackedModel*>(static_cast<intptr_t>(handle));
    return m ? static_cast<jlong>(m->param_count()) : 0;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativePackedConfig(
    JNIEnv* env, jobject /*thiz*/, jlong handle) {
    auto* m = reinterpret_cast<PackedModel*>(static_cast<intptr_t>(handle));
    if (!m) return nullptr;
    const ModelConfig& c = m->config();
    jint vals[9] = {c.vocab, c.context, c.d_model, c.layers, c.heads, c.d_ff,
                    c.tied_embeddings ? 1 : 0, m->bits(), m->group()};
    jintArray result = env->NewIntArray(9);
    if (result) env->SetIntArrayRegion(result, 0, 9, vals);
    return result;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_modelforge_app_nativebridge_NativeBridge_nativePackedGenerate(
    JNIEnv* env, jobject /*thiz*/, jlong handle, jintArray prompt,
    jint max_new, jfloat temperature, jint top_k, jfloat top_p, jlong seed) {
    auto* m = reinterpret_cast<PackedModel*>(static_cast<intptr_t>(handle));
    if (!m) return nullptr;
    try {
        std::vector<int> p;
        if (!read_ints(env, prompt, p)) return nullptr;
        for (int& v : p) {
            if (v < 0) v = 0;
            if (v >= m->config().vocab) v = m->config().vocab - 1;
        }
        const auto out = m->generate(p, max_new, temperature, top_k, top_p,
                                     static_cast<uint64_t>(seed));
        jintArray result = env->NewIntArray(static_cast<jsize>(out.size()));
        if (result && !out.empty()) {
            env->SetIntArrayRegion(result, 0, static_cast<jsize>(out.size()),
                                   reinterpret_cast<const jint*>(out.data()));
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* /*vm*/, void* /*reserved*/) {
    return JNI_VERSION_1_6;
}
