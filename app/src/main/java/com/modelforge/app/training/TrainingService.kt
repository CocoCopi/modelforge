package com.modelforge.app.training

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.modelforge.app.MainActivity
import com.modelforge.app.R
import com.modelforge.app.core.HardwareDetector
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingStatus
import com.modelforge.app.data.tokenizer.BpeTokenizer
import com.modelforge.app.nativebridge.NativeModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Random
import kotlin.math.max
import kotlin.math.min

class TrainingService : Service() {

    companion object {
        const val ACTION_START = "com.modelforge.app.action.START_TRAINING"
        const val ACTION_PAUSE = "com.modelforge.app.action.PAUSE_TRAINING"
        const val ACTION_RESUME = "com.modelforge.app.action.RESUME_TRAINING"
        const val ACTION_STOP = "com.modelforge.app.action.STOP_TRAINING"
        const val EXTRA_MODEL_ID = "model_id"

        private const val CHANNEL_ID = "modelforge_training"
        private const val NOTIFICATION_ID = 1001

        fun start(context: Context, modelId: String) {
            val intent = Intent(context, TrainingService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_MODEL_ID, modelId)
            ContextCompat.startForegroundService(context, intent)
        }

        fun pause(context: Context) = sendAction(context, ACTION_PAUSE)
        fun resume(context: Context) = sendAction(context, ACTION_RESUME)
        fun stop(context: Context) = sendAction(context, ACTION_STOP)

        private fun sendAction(context: Context, action: String) {
            val intent = Intent(context, TrainingService::class.java).setAction(action)
            runCatching { context.startService(intent) }
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var store: ModelStore
    private var job: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var currentModelId: String? = null
    private var modelName: String = ""
    private var currentState: TrainingLiveState? = null

    @Volatile private var paused = false
    @Volatile private var autoPaused = false
    @Volatile private var stopRequested = false

    private var lastNotifyAt = 0L
    private var lastPersistAt = 0L

    override fun onCreate() {
        super.onCreate()
        store = ModelStore(this)
        createChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        when (action) {
            ACTION_PAUSE -> {
                pauseTraining(manual = true)
                if (job == null) stopSelf()
            }
            ACTION_RESUME -> {
                resumeTraining()
                if (job == null) stopSelf()
            }
            ACTION_STOP -> {
                stopTraining()
                if (job == null) stopSelf()
            }
            ACTION_START -> {
                val id = intent.getStringExtra(EXTRA_MODEL_ID)
                if (id != null) beginTraining(id)
            }
            null -> {
                val active = store.activeModelId()
                if (active != null) {
                    beginTraining(active)
                } else {
                    stopSelf()
                }
            }
        }
        return START_STICKY
    }

    private fun beginTraining(modelId: String) {
        if (job?.isActive == true) {
            if (currentModelId == modelId) return
            stopRequested = true
            job?.cancel()
        }
        val manifest = store.loadManifest(modelId)
        modelName = manifest?.name ?: modelId
        currentModelId = modelId
        stopRequested = false
        paused = false
        autoPaused = false
        store.setActive(modelId, true)
        val initial = TrainingLiveState(
            modelId = modelId,
            status = TrainingStatus.PREPARING,
            message = "Preparing tokenizer and dataset…",
            totalSteps = manifest?.settings?.steps ?: 0,
            batchSize = manifest?.settings?.batchSize ?: 0
        )
        publishState(initial, persist = true, forceNotify = true)
        startInForeground(initial)
        job = scope.launch {
            runTraining(modelId)
        }
    }

    private fun pauseTraining(manual: Boolean) {
        paused = true
        autoPaused = false
        releaseWakeLock()
        val state = currentState ?: return
        val updated = if (manual) {
            state.copy(
                status = TrainingStatus.PAUSED,
                message = "Paused by user.",
                autoPaused = false,
                updatedAt = System.currentTimeMillis()
            )
        } else {
            state.copy(
                status = TrainingStatus.PAUSED,
                message = "Paused: device conditions.",
                updatedAt = System.currentTimeMillis()
            )
        }
        publishState(updated, persist = true, forceNotify = true)
    }

    private fun resumeTraining() {
        paused = false
        autoPaused = false
        acquireWakeLock()
        val state = currentState ?: return
        publishState(
            state.copy(
                status = TrainingStatus.RUNNING,
                message = "Resumed by user.",
                autoPaused = false,
                updatedAt = System.currentTimeMillis()
            ),
            persist = true,
            forceNotify = true
        )
    }

    private fun stopTraining() {
        stopRequested = true
        paused = false
        autoPaused = false
        currentState?.let { state ->
            publishState(
                state.copy(message = "Stopping and saving a checkpoint…", updatedAt = System.currentTimeMillis()),
                persist = true,
                forceNotify = true
            )
        }
    }

    private suspend fun runTraining(modelId: String) {
        var model: NativeModel? = null
        try {
            val manifest = store.loadManifest(modelId)
                ?: throw TrainingFailure(
                    "Model files were not found.",
                    "Return to Home and open the model again, or create a new model."
                )
            modelName = manifest.name
            val cfg = manifest.config
            val settings = manifest.settings
            val totalSteps = settings.steps.coerceAtLeast(1)
            val checkpointEvery = settings.checkpointEvery.coerceAtLeast(1)
            val sampleEvery = settings.sampleEvery.coerceAtLeast(1)
            val seq = min(settings.seqLength, cfg.contextLength).coerceAtLeast(8)
            val previous = store.loadLiveState(modelId)
            var state = TrainingLiveState(
                modelId = modelId,
                status = TrainingStatus.PREPARING,
                message = "Loading dataset cache…",
                totalSteps = totalSteps,
                batchSize = settings.batchSize,
                bestLoss = previous?.bestLoss ?: Float.MAX_VALUE,
                lossHistory = previous?.lossHistory ?: emptyList(),
                elapsedMs = previous?.elapsedMs ?: 0L
            )
            publishState(state, persist = true, forceNotify = true)

            if (manifest.datasets.isEmpty()) {
                throw TrainingFailure(
                    "No datasets are attached to this model.",
                    "Add at least one dataset in Step 3 before training."
                )
            }

            val dataDir = store.dataDir(modelId)
            val corpusCap = settings.corpusChars.coerceIn(10_000, 4_000_000)
            val identityBudget = if (manifest.creator.isNotBlank()) {
                IdentityDataGenerator.estimateIdentityFraction(
                    corpusCap,
                    settings.identityFraction.toDouble()
                )
            } else {
                0
            }
            val corpus = withContext(Dispatchers.IO) {
                DatasetMixer.buildWeightedCorpus(
                    manifest.datasets,
                    dataDir,
                    (corpusCap - identityBudget - 2).coerceAtLeast(8_000)
                )
            }
            if (corpus.text.isBlank()) {
                throw TrainingFailure(
                    "The cached dataset text is empty.",
                    "Re-open Step 3 and download the dataset again."
                )
            }
            val identity = if (identityBudget > 0) {
                IdentityDataGenerator.generate(
                    name = manifest.name,
                    creator = manifest.creator,
                    languages = manifest.languages,
                    maxChars = identityBudget
                )
            } else {
                null
            }
            val trainingText = if (identity == null || identity.text.isBlank()) {
                corpus.text
            } else {
                corpus.text + "\n\n" + identity.text
            }

            state = state.copy(
                message = "Training tokenizer on ${trainingText.length} characters " +
                    "(${identity?.conversations?.size ?: 0} identity conversations)…"
            )
            publishState(state, persist = false, forceNotify = true)

            val tokenizer = loadOrTrainTokenizer(manifest, trainingText)
            state = state.copy(message = "Tokenizing dataset…")
            publishState(state, persist = false, forceNotify = true)

            val tokens = withContext(Dispatchers.Default) {
                getOrBuildTokens(manifest, tokenizer, trainingText, state)
            }
            if (tokens.size < seq + 2) {
                throw TrainingFailure(
                    "Not enough text for a sequence of $seq tokens (${tokens.size} tokens).",
                    "Download more rows in Step 3 or lower the sequence length."
                )
            }
            val maxToken = tokens.maxOrNull() ?: 0
            if (maxToken >= cfg.vocabSize) {
                throw TrainingFailure(
                    "The tokenizer produced token $maxToken but the model vocabulary is ${cfg.vocabSize}.",
                    "Delete this model's tokenizer.json and start training again to rebuild it."
                )
            }

            state = state.copy(message = "Allocating model…")
            publishState(state, persist = false, forceNotify = true)

            model = openModel(manifest)
            var step = model!!.step.toInt().coerceAtLeast(0)
            if (step >= totalSteps) {
                finish(
                    modelId,
                    TrainingStatus.COMPLETED,
                    "Training was already complete at step $step.",
                    state
                )
                model!!.close()
                return
            }

            var batch = settings.batchSize
            val adjustment = MemoryGuard.adjustForDevice(cfg, batch, seq, this)
            batch = adjustment.batchSize
            var memoryMessage = adjustment.message

            acquireWakeLock()
            state = state.copy(
                status = TrainingStatus.RUNNING,
                message = memoryMessage ?: "Training.",
                step = step,
                batchSize = batch
            )
            publishState(state, persist = true, forceNotify = true)

            val sampleTokens = batch * seq
            val inputTokens = IntArray(sampleTokens)
            val targetTokens = IntArray(sampleTokens)
            val random = Random(System.nanoTime())
            var bestLoss = state.bestLoss
            val history = state.lossHistory.toMutableList()
            var elapsedMs = state.elapsedMs
            var sample = state.sample
            var lastDeviceCheck = 0L
            var lastNotify = 0L

            while (step < totalSteps) {
                if (stopRequested) break

                if (paused) {
                    if (autoPaused) {
                        val reason = devicePauseReason()
                        if (reason == null) {
                            paused = false
                            autoPaused = false
                            acquireWakeLock()
                            state = state.copy(
                                status = TrainingStatus.RUNNING,
                                message = "Resumed automatically: battery and temperature are safe again.",
                                autoPaused = false,
                                updatedAt = System.currentTimeMillis()
                            )
                            publishState(state, persist = true, forceNotify = true)
                        } else {
                            val now = System.currentTimeMillis()
                            state = state.copy(
                                status = TrainingStatus.PAUSED,
                                message = reason,
                                autoPaused = true,
                                batteryPercent = HardwareDetector.batteryPercent(this),
                                thermalStatus = HardwareDetector.thermalStatus(this),
                                deviceTempC = HardwareDetector.batteryTemperatureC(this),
                                ramMb = HardwareDetector.processRssMb(),
                                updatedAt = now
                            )
                            publishState(
                                state,
                                persist = now - lastPersistAt > 10_000,
                                notify = now - lastNotify > 5_000
                            )
                            delay(10_000)
                        }
                    } else {
                        delay(500)
                    }
                    continue
                }

                val now = System.currentTimeMillis()
                if (now - lastDeviceCheck > 5_000 || step % 5 == 0) {
                    lastDeviceCheck = now
                    val reason = devicePauseReason()
                    if (reason != null) {
                        paused = true
                        autoPaused = true
                        releaseWakeLock()
                        state = state.copy(
                            status = TrainingStatus.PAUSED,
                            message = reason,
                            autoPaused = true,
                            batteryPercent = HardwareDetector.batteryPercent(this),
                            thermalStatus = HardwareDetector.thermalStatus(this),
                            deviceTempC = HardwareDetector.batteryTemperatureC(this),
                            updatedAt = now
                        )
                        publishState(state, persist = true, forceNotify = true)
                        continue
                    }
                }

                val lr = LrSchedule.learningRate(step, totalSteps, settings.warmupSteps, settings.learningRate)
                fillBatch(tokens, inputTokens, targetTokens, batch, seq, random)

                val t0 = System.nanoTime()
                val loss = model!!.trainStep(inputTokens, targetTokens, batch, seq, lr, settings.weightDecay)
                val seconds = (System.nanoTime() - t0) / 1_000_000_000.0

                if (!loss.isFinite() || loss < 0f) {
                    val oldBatch = batch
                    batch /= 2
                    if (batch < 1) batch = 1
                    if (oldBatch == 1) {
                        throw TrainingFailure(
                            "The model ran out of memory even with batch size 1.",
                            "Close other apps, use a smaller model, or lower the sequence length."
                        )
                    }
                    model!!.close()
                    model = openModel(manifest)
                    step = model!!.step.toInt().coerceAtLeast(0)
                    memoryMessage = "Low memory: batch size lowered from $oldBatch to $batch. " +
                        "Training resumed from step $step."
                    state = state.copy(
                        batchSize = batch,
                        step = step,
                        message = memoryMessage,
                        updatedAt = System.currentTimeMillis()
                    )
                    publishState(state, persist = true, forceNotify = true)
                    continue
                }

                step++
                elapsedMs += (seconds * 1000.0).toLong()
                val tokensPerSec = if (seconds > 0) sampleTokens / seconds else 0.0
                val eta = if (tokensPerSec > 0) {
                    ((totalSteps - step) * seconds).toLong()
                } else 0L
                if (loss < bestLoss) bestLoss = loss
                history.add(loss)
                while (history.size > 300) history.removeAt(0)

                if (step % sampleEvery == 0 || step == totalSteps) {
                    sample = generateSample(model!!, tokenizer, settings.samplingTemperature)
                }

                state = state.copy(
                    status = TrainingStatus.RUNNING,
                    message = memoryMessage ?: "Training.",
                    step = step,
                    loss = loss,
                    bestLoss = bestLoss,
                    tokensPerSec = tokensPerSec,
                    etaSeconds = eta,
                    ramMb = HardwareDetector.processRssMb(),
                    batteryPercent = HardwareDetector.batteryPercent(this),
                    thermalStatus = HardwareDetector.thermalStatus(this),
                    deviceTempC = HardwareDetector.batteryTemperatureC(this),
                    sample = sample,
                    lossHistory = history.toList(),
                    elapsedMs = elapsedMs,
                    batchSize = batch,
                    autoPaused = false,
                    updatedAt = System.currentTimeMillis()
                )

                if (step % checkpointEvery == 0 || step == totalSteps || stopRequested) {
                    val last = store.lastCheckpoint(modelId)
                    val ok = model!!.saveCheckpoint(last.absolutePath, true)
                    if (ok && loss <= bestLoss + 1e-6f) {
                        model!!.saveCheckpoint(store.bestCheckpoint(modelId).absolutePath, false)
                    }
                    store.saveLiveState(state)
                    lastPersistAt = System.currentTimeMillis()
                    if (!ok) {
                        state = state.copy(
                            message = (memoryMessage ?: "") + " Warning: checkpoint could not be written."
                        )
                    }
                }

                if (now - lastNotify > 1_000) {
                    lastNotify = now
                    publishState(state, persist = false, notify = true)
                } else {
                    publishState(state, persist = false, notify = false)
                }
            }

            if (model!!.isOpen) {
                model!!.saveCheckpoint(store.lastCheckpoint(modelId).absolutePath, true)
            }
            val status = if (stopRequested) TrainingStatus.STOPPED else TrainingStatus.COMPLETED
            val message = if (stopRequested) {
                "Training stopped at step $step. The checkpoint is saved."
            } else {
                "Training finished at step $step. Best measured loss ${String.format("%.4f", bestLoss)}."
            }
            finish(modelId, status, message, state.copy(step = step, totalSteps = totalSteps, bestLoss = bestLoss))
        } catch (t: TrainingFailure) {
            fail(modelId, t.message ?: "Training failed.", t.suggestion)
        } catch (t: Throwable) {
            val message = t.message ?: t.javaClass.simpleName
            fail(
                modelId,
                "Training stopped: $message",
                "Try again. If this repeats, close other apps, lower the batch size or create a smaller model."
            )
        } finally {
            runCatching { model?.close() }
            releaseWakeLock()
            job = null
        }
    }

    private suspend fun loadOrTrainTokenizer(manifest: ModelManifest, corpus: String): BpeTokenizer {
        val file = store.tokenizerFile(manifest.id)
        if (file.exists()) {
            val loaded = runCatching { BpeTokenizer.load(file) }.getOrNull()
            if (loaded != null && loaded.vocabSize <= manifest.config.vocabSize) {
                return loaded
            }
        }
        val tokenizer = withContext(Dispatchers.Default) {
            BpeTokenizer.train(
                corpus,
                manifest.config.vocabSize,
                maxTrainingChars = min(corpus.length, manifest.settings.corpusChars)
            ) { done, total ->
                if (done % 100 == 0 || done == total) {
                    val now = System.currentTimeMillis()
                    val state = currentState
                    if (state != null && now - lastNotifyAt > 1_000) {
                        lastNotifyAt = now
                        val updated = state.copy(
                            message = "Training tokenizer: $done / $total merges",
                            updatedAt = now
                        )
                        currentState = updated
                        TrainingBus.publish(updated)
                        updateNotification(updated)
                    }
                }
            }
        }
        tokenizer.save(file)
        store.saveManifest(manifest.copy(tokenizerTrained = true))
        return tokenizer
    }

    private suspend fun getOrBuildTokens(
        manifest: ModelManifest,
        tokenizer: BpeTokenizer,
        corpus: String,
        state: TrainingLiveState
    ): IntArray {
        val cacheFile = store.tokenCacheFile(manifest.id)
        val cacheManifest = store.tokenCacheManifestFile(manifest.id)
        val signature = TokenCache.signature(
            store.tokenizerFile(manifest.id),
            manifest.datasets,
            store.dataDir(manifest.id),
            manifest.config.vocabSize,
            identityKey = listOf(
                manifest.name,
                manifest.creator,
                manifest.languages.joinToString(","),
                manifest.behavior.reasoningEnabled.toString(),
                manifest.behavior.toolCalling.toString(),
                manifest.settings.identityFraction.toString()
            ).joinToString("|")
        )
        TokenCache.load(cacheFile, cacheManifest, signature)?.let { return it }

        val tokens = withContext(Dispatchers.Default) {
            currentCoroutineContext().ensureActive()
            tokenizer.encode(corpus)
        }
        TokenCache.save(cacheFile, cacheManifest, tokens, signature, tokenizer.vocabSize)
        return tokens
    }

    private fun openModel(manifest: ModelManifest): NativeModel {
        val last = store.lastCheckpoint(manifest.id)
        val cfg = manifest.config
        val model = if (last.exists()) {
            NativeModel.load(last.absolutePath, withOptimizer = true)
                ?: throw TrainingFailure(
                    "The checkpoint could not be loaded.",
                    "Delete checkpoints/last.mfck for this model and start training again."
                )
        } else {
            NativeModel.create(cfg, System.nanoTime())
                ?: throw TrainingFailure(
                    "The model could not be allocated.",
                    "Close other apps or create a smaller model."
                )
        }
        val c = model.config
        if (c.vocabSize != cfg.vocabSize || c.contextLength != cfg.contextLength ||
            c.dModel != cfg.dModel || c.layers != cfg.layers || c.heads != cfg.heads ||
            c.dFf != cfg.dFf || c.tiedEmbeddings != cfg.tiedEmbeddings
        ) {
            model.close()
            throw TrainingFailure(
                "The checkpoint architecture does not match this model's config.",
                "Return to Home, delete the broken model and create it again."
            )
        }
        if (cfg.bits in 2..8) {
            model.setQuantization(cfg.bits, 32)
        }
        if (!last.exists()) {
            if (!model.saveCheckpoint(last.absolutePath, true)) {
                model.close()
                throw TrainingFailure(
                    "Could not write the initial checkpoint.",
                    "Check that the phone has free storage."
                )
            }
        }
        return model
    }

    private fun fillBatch(
        data: IntArray,
        tokens: IntArray,
        targets: IntArray,
        batch: Int,
        seq: Int,
        random: Random
    ) {
        val maxStart = data.size - seq - 1
        require(maxStart >= 0)
        for (b in 0 until batch) {
            val start = random.nextInt(maxStart + 1)
            val base = b * seq
            for (t in 0 until seq) {
                tokens[base + t] = data[start + t]
                targets[base + t] = data[start + t + 1]
            }
        }
    }

    private fun generateSample(
        model: NativeModel,
        tokenizer: BpeTokenizer,
        temperature: Float
    ): String {
        return runCatching {
            val prompt = tokenizer.encode("Once upon a time")
            val generated = model.generate(
                prompt = prompt,
                maxNew = 32,
                temperature = temperature,
                topK = 40,
                topP = 0.9f,
                seed = System.nanoTime()
            )
            tokenizer.decode(prompt + generated)
        }.getOrDefault("")
    }

    private fun devicePauseReason(): String? {
        val battery = HardwareDetector.batteryPercent(this)
        if (battery in 1..14) {
            return "Battery ${battery}% is below 15%. Training is paused and will resume automatically once charged."
        }
        if (HardwareDetector.shouldPauseForHeat(this)) {
            return "Phone temperature is ${HardwareDetector.thermalStatus(this)}. Training is paused until it cools down."
        }
        return null
    }

    private fun finish(modelId: String, status: String, message: String, state: TrainingLiveState) {
        store.setActive(modelId, false)
        val finalState = state.copy(
            status = status,
            message = message,
            autoPaused = false,
            batteryPercent = HardwareDetector.batteryPercent(this),
            thermalStatus = HardwareDetector.thermalStatus(this),
            deviceTempC = HardwareDetector.batteryTemperatureC(this),
            ramMb = HardwareDetector.processRssMb(),
            updatedAt = System.currentTimeMillis()
        )
        publishState(finalState, persist = true, forceNotify = true)
        releaseWakeLock()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    private fun fail(modelId: String, message: String, suggestion: String?) {
        store.setActive(modelId, false)
        val combined = if (suggestion.isNullOrBlank()) message else "$message Suggested fix: $suggestion"
        val state = (currentState ?: TrainingLiveState(modelId, TrainingStatus.ERROR)).copy(
            status = TrainingStatus.ERROR,
            message = combined,
            error = combined,
            autoPaused = false,
            updatedAt = System.currentTimeMillis()
        )
        publishState(state, persist = true, forceNotify = true)
        releaseWakeLock()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun publishState(
        state: TrainingLiveState,
        persist: Boolean = false,
        notify: Boolean = true,
        forceNotify: Boolean = false
    ) {
        currentState = state
        TrainingBus.publish(state)
        if (persist) {
            runCatching { store.saveLiveState(state) }
            lastPersistAt = System.currentTimeMillis()
        }
        val now = System.currentTimeMillis()
        if (notify || forceNotify) {
            if (forceNotify || now - lastNotifyAt >= 1_000) {
                lastNotifyAt = now
                updateNotification(state)
            }
        }
    }

    private fun updateNotification(state: TrainingLiveState) {
        val manager = NotificationManagerCompat.from(this)
        runCatching { manager.notify(NOTIFICATION_ID, buildNotification(state)) }
    }

    private fun startInForeground(state: TrainingLiveState) {
        val notification = buildNotification(state)
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        } else {
            0
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
    }

    private fun buildNotification(state: TrainingLiveState): Notification {
        val pausedNow = state.status == TrainingStatus.PAUSED
        val title = when (state.status) {
            TrainingStatus.PREPARING -> "Preparing $modelName"
            TrainingStatus.COMPLETED -> "$modelName training complete"
            TrainingStatus.STOPPED -> "$modelName training stopped"
            TrainingStatus.ERROR -> "$modelName training failed"
            TrainingStatus.PAUSED -> "$modelName paused"
            else -> "Training $modelName"
        }
        val text = buildString {
            if (state.totalSteps > 0) append("Step ${state.step}/${state.totalSteps}")
            if (state.loss > 0f) append(" • loss ${String.format("%.3f", state.loss)}")
            if (state.tokensPerSec > 1.0) append(" • ${String.format("%.0f", state.tokensPerSec)} tok/s")
            if (state.ramMb > 0) append(" • ${state.ramMb} MB RAM")
            if (state.batteryPercent >= 0) append(" • ${state.batteryPercent}%")
        }.ifBlank { state.message }

        val openIntent = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        val openPending = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val toggleAction = if (pausedNow) ACTION_RESUME else ACTION_PAUSE
        val toggleLabel = if (pausedNow) "Resume" else "Pause"
        val togglePending = PendingIntent.getService(
            this, 1,
            Intent(this, TrainingService::class.java).setAction(toggleAction),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopPending = PendingIntent.getService(
            this, 2,
            Intent(this, TrainingService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(state.message.ifBlank { text }))
            .setContentIntent(openPending)
            .setOnlyAlertOnce(true)
            .setOngoing(state.isActive)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)

        if (state.status == TrainingStatus.PREPARING || state.totalSteps <= 0) {
            builder.setProgress(0, 0, true)
        } else {
            val used = (state.step * 1000 / max(1, state.totalSteps)).coerceIn(0, 1000)
            builder.setProgress(1000, used, false)
            builder.setSubText(
                buildString {
                    if (state.tokensPerSec > 0) {
                        val eta = state.etaSeconds
                        append("ETA ${formatDuration(eta)}")
                    }
                    if (state.thermalStatus.isNotBlank()) {
                        if (isNotEmpty()) append(" • ")
                        append("thermal ${state.thermalStatus}")
                    }
                    state.deviceTempC?.let {
                        if (isNotEmpty()) append(" • ")
                        append(String.format("%.1f°C", it))
                    }
                }
            )
        }

        if (state.isActive) {
            builder.addAction(R.drawable.ic_notification, toggleLabel, togglePending)
            builder.addAction(R.drawable.ic_notification, "Stop", stopPending)
        }
        return builder.build()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.training_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.training_channel_description)
                setShowBadge(false)
            }
            manager.createNotificationChannel(channel)
        }
    }

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock() {
        if (wakeLock == null) {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ModelForge:Training").apply {
                setReferenceCounted(false)
            }
        }
        val lock = wakeLock ?: return
        if (!lock.isHeld) {
            runCatching { lock.acquire() }
        }
    }

    private fun releaseWakeLock() {
        val lock = wakeLock ?: return
        if (lock.isHeld) runCatching { lock.release() }
    }

    override fun onDestroy() {
        stopRequested = true
        releaseWakeLock()
        scope.cancel()
        super.onDestroy()
    }
}

private fun formatDuration(seconds: Long): String {
    if (seconds <= 0) return "—"
    val h = seconds / 3600
    val m = (seconds % 3600) / 60
    val s = seconds % 60
    return when {
        h > 0 -> "${h}h ${m}m"
        m > 0 -> "${m}m ${s}s"
        else -> "${s}s"
    }
}

private class TrainingFailure(message: String, val suggestion: String? = null) : Exception(message)
