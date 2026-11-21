package com.modelforge.app.training

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.AlarmManager
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.SystemClock
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
import com.modelforge.app.data.ModelCard
import com.modelforge.app.data.ModelManifest
import com.modelforge.app.data.ModelStore
import com.modelforge.app.data.TrainingLiveState
import com.modelforge.app.data.TrainingStatus
import com.modelforge.app.data.tokenizer.BpeTokenizer
import com.modelforge.app.nativebridge.NativeModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
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
        const val EXTRA_EXTEND = "extend_training"

        private const val CHANNEL_ID = "modelforge_training"
        private const val NOTIFICATION_ID = 1001

        fun start(context: Context, modelId: String, extend: Boolean = false) {
            val intent = Intent(context, TrainingService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_MODEL_ID, modelId)
                .putExtra(EXTRA_EXTEND, extend)
            ContextCompat.startForegroundService(context, intent)
        }

        fun pause(context: Context) = sendAction(context, ACTION_PAUSE)
        fun resume(context: Context) = sendAction(context, ACTION_RESUME)
        fun stop(context: Context) = sendAction(context, ACTION_STOP)

        private fun sendAction(context: Context, action: String) {
            // Starting a background service is illegal on API 26+ unless the
            // app is in a foreground state. The service is already foreground
            // while training, so these actions only need to reach a live run.
            if (ModelStore(context).activeModelId() == null) return
            val intent = Intent(context, TrainingService::class.java).setAction(action)
            // The service is already running, so a plain start is accepted
            // even when the app is in the background (the notification action).
            // startForegroundService is only the fallback for a service that
            // is not running yet.
            val delivered = runCatching { context.startService(intent) }.isSuccess
            if (!delivered) {
                runCatching { ContextCompat.startForegroundService(context, intent) }
            }
        }
    }

    private sealed interface Command {
        data class Start(val modelId: String, val extend: Boolean) : Command
        data object Pause : Command
        data object Resume : Command
        data object Stop : Command
    }

    /** Per-run control block. A new run never reads the previous run's flags. */
    private class RunControl(val extend: Boolean) {
        @Volatile var stopRequested = false
        @Volatile var paused = false
        @Volatile var autoPaused = false
        @Volatile var checkpointRequested = false
        @Volatile var resumedWithRecovery = false
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var store: ModelStore
    private val commands = Channel<Command>(Channel.UNLIMITED)
    private var job: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var currentModelId: String? = null
    private var modelName: String = ""
    @Volatile private var currentState: TrainingLiveState? = null
    private var control = RunControl(extend = false)

    /**
     * The model owned by the current run. Stop reaches through this to abort a
     * native step that is already executing; it is cleared before the model is
     * freed.
     */
    @Volatile private var stopTarget: NativeModel? = null

    private var lastNotifyAt = 0L
    private var lastPersistAt = 0L

    /**
     * Every service command is applied by one coroutine in arrival order. The
     * training loop is the only other writer, so there is no window in which a
     * stale job can reset a new job's flags or null its reference.
     */
    private val commandJob: Job = scope.launch {
        for (command in commands) {
            when (command) {
                is Command.Start -> handleStart(command)
                Command.Pause -> pauseTraining(manual = true)
                Command.Resume -> resumeTraining()
                Command.Stop -> stopTraining()
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        store = ModelStore(this)
        createChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_PAUSE -> commands.trySend(Command.Pause)
            ACTION_RESUME -> commands.trySend(Command.Resume)
            ACTION_STOP -> commands.trySend(Command.Stop)
            ACTION_START -> {
                val id = intent.getStringExtra(EXTRA_MODEL_ID)
                if (id != null) {
                    commands.trySend(
                        Command.Start(id, intent.getBooleanExtra(EXTRA_EXTEND, false))
                    )
                } else {
                    stopSelf()
                }
            }
            null -> {
                val active = store.activeModelId()
                val live = active?.let { store.loadLiveState(it) }
                // A process restart must resume an interrupted run, but it must
                // never resurrect a run the user stopped (or one that already
                // finished or failed). The stop path clears the active marker,
                // and this guard covers a marker left behind by a crash.
                val terminal = live != null && (
                    live.status == TrainingStatus.COMPLETED ||
                        live.status == TrainingStatus.STOPPED ||
                        live.status == TrainingStatus.ERROR
                    )
                if (active != null && !terminal &&
                    store.loadManifest(active)?.settings?.autoResume != false
                ) {
                    commands.trySend(Command.Start(active, false))
                } else {
                    if (active != null && terminal) store.setActive(active, false)
                    stopSelf()
                }
            }
        }
        return START_STICKY
    }

    private suspend fun handleStart(command: Command.Start) {
        val existing = job
        if (existing?.isActive == true) {
            if (currentModelId == command.modelId) return
            control.stopRequested = true
            // Abort the old native step before waiting for the job to unwind,
            // otherwise switching models can block for a whole step.
            stopTarget?.requestStop()
            existing.cancelAndJoin()
        }
        val manifest = store.loadManifest(command.modelId)
        if (manifest == null) {
            stopSelf()
            return
        }
        modelName = manifest.name
        currentModelId = command.modelId
        control = RunControl(command.extend)
        store.setActive(command.modelId, true)
        val initial = TrainingLiveState(
            modelId = command.modelId,
            status = TrainingStatus.PREPARING,
            message = "Preparing tokenizer and dataset…",
            totalSteps = manifest.settings.steps.coerceAtLeast(1),
            batchSize = manifest.settings.batchSize
        )
        publishState(initial, persist = true, forceNotify = true)
        startInForeground(initial)
        val run = control
        job = scope.launch {
            runTraining(command.modelId, run)
        }
    }

    private fun pauseTraining(manual: Boolean) {
        control.paused = true
        control.autoPaused = false
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
        control.paused = false
        control.autoPaused = false
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
        control.stopRequested = true
        control.paused = false
        control.autoPaused = false
        // Reach into the native engine right away. A big model can spend tens
        // of seconds inside a single step, so waiting for the loop to notice
        // the flag is what made "Stopping…" look like a permanent hang. The
        // engine aborts before AdamW, so the last checkpoint's weights stay
        // consistent.
        stopTarget?.requestStop()
        // Drop the active marker now rather than in finish(): if the process
        // dies while the final checkpoint is being written, the next launch
        // must not auto-resume a run the user stopped.
        currentModelId?.let { store.setActive(it, false) }
        currentState?.let { state ->
            publishState(
                state.copy(
                    status = TrainingStatus.STOPPED,
                    message = "Stopping now. Saving the latest weights…",
                    updatedAt = System.currentTimeMillis()
                ),
                persist = true,
                forceNotify = true
            )
        }
    }

    private suspend fun runTraining(modelId: String, run: RunControl) {
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
            var totalSteps = settings.steps.coerceAtLeast(1)
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
            // The old 4M-character clamp silently discarded most of a large
            // dataset, so every epoch replayed the same small slice. Scale the
            // budget to the step count and to available RAM instead.
            val corpusCap = CorpusBudget.resolve(
                requested = settings.corpusChars,
                steps = totalSteps,
                batchSize = settings.batchSize,
                seqLength = seq,
                availableRamBytes = HardwareDetector.availableRamBytes(this)
            )
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
            if (run.stopRequested) throw TrainingStopped()
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

            val tokenizer = loadOrTrainTokenizer(manifest, trainingText, corpusCap)
            if (run.stopRequested) throw TrainingStopped()
            state = state.copy(message = "Tokenizing dataset…")
            publishState(state, persist = false, forceNotify = true)

            val tokens = withContext(Dispatchers.Default) {
                getOrBuildTokens(manifest, tokenizer, trainingText, state)
            }
            if (run.stopRequested) throw TrainingStopped()
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

            var memoryMessage: String? = null
            model = openModel(manifest, run)
            // A fresh engine must not inherit a stop that was meant for the
            // previous run (for example after switching models).
            model!!.clearStop()
            stopTarget = model
            if (run.stopRequested) throw TrainingStopped()
            var step = model!!.step.toInt().coerceAtLeast(0)
            if (run.resumedWithRecovery && step > 0) {
                memoryMessage = "Recovered the latest weights from step $step after the app " +
                    "was closed. Optimizer state may restart; progress is preserved."
            }
            // "Train again" on a finished model extends the run instead of
            // reporting that there is nothing to do.
            if (run.extend && step >= totalSteps) {
                totalSteps = step + settings.steps.coerceAtLeast(1)
                state = state.copy(totalSteps = totalSteps)
            }
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
            if (adjustment.message != null) {
                memoryMessage = if (memoryMessage == null) {
                    adjustment.message
                } else {
                    "$memoryMessage ${adjustment.message}"
                }
            }

            acquireWakeLock()
            state = state.copy(
                status = TrainingStatus.RUNNING,
                message = memoryMessage ?: "Training.",
                step = step,
                batchSize = batch
            )
            publishState(state, persist = true, forceNotify = true)

            val recoveryEvery = RecoveryPolicy.intervalFor(cfg.paramCount())
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
                // Cancellation means a new run has taken over; it must not be
                // treated as a failure of this one.
                currentCoroutineContext().ensureActive()
                if (run.stopRequested) break

                if (run.paused) {
                    if (run.autoPaused) {
                        val reason = devicePauseReason()
                        if (reason == null) {
                            run.paused = false
                            run.autoPaused = false
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
                        while (run.paused && !run.stopRequested) {
                            currentCoroutineContext().ensureActive()
                            delay(250)
                        }
                    }
                    continue
                }

                val now = System.currentTimeMillis()
                if (now - lastDeviceCheck > 5_000 || step % 5 == 0) {
                    lastDeviceCheck = now
                    val reason = devicePauseReason()
                    if (reason != null) {
                        run.paused = true
                        run.autoPaused = true
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

                if (loss == NativeModel.STOPPED) {
                    // The engine unwound mid-step at our request. AdamW did not
                    // run, so the weights still match the last checkpoint.
                    run.stopRequested = true
                    break
                }

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
                    model = openModel(manifest, run)
                    model!!.clearStop()
                    stopTarget = model
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

                val recoveryDue = RecoveryPolicy.isDue(step, totalSteps, recoveryEvery) ||
                    run.checkpointRequested
                if (recoveryDue) {
                    run.checkpointRequested = false
                    withContext(Dispatchers.IO) {
                        model!!.saveCheckpoint(store.recoveryCheckpoint(modelId).absolutePath, false)
                    }
                }

                if (step % checkpointEvery == 0 || step == totalSteps || run.stopRequested) {
                    val last = store.lastCheckpoint(modelId)
                    val ok = withContext(Dispatchers.IO) {
                        model!!.saveCheckpoint(last.absolutePath, true)
                    }
                    if (ok && loss <= bestLoss + 1e-6f) {
                        withContext(Dispatchers.IO) {
                            model!!.saveCheckpoint(store.bestCheckpoint(modelId).absolutePath, false)
                        }
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
                    publishState(state, persist = true, notify = true)
                } else {
                    publishState(state, persist = false, notify = false)
                }
            }

            // Save the final weights and make sure a loadable "best" checkpoint
            // exists. The playground, identity check and exporters all read
            // best.mfck; a run stopped before its first periodic checkpoint
            // used to announce success while that file had never been written.
            var savedOk = true
            if (model!!.isOpen) {
                savedOk = withContext(Dispatchers.IO) {
                    val lastOk = model!!.saveCheckpoint(
                        store.lastCheckpoint(modelId).absolutePath,
                        true
                    )
                    val best = store.bestCheckpoint(modelId)
                    val bestOk = if (CheckpointHeader.hasMagic(best)) {
                        true
                    } else {
                        model!!.saveCheckpoint(best.absolutePath, false)
                    }
                    lastOk && bestOk
                }
            }
            val stopped = run.stopRequested
            val status = if (stopped) TrainingStatus.STOPPED else TrainingStatus.COMPLETED
            val verb = if (stopped) "stopped" else "finished"
            val message = when {
                !savedOk -> "Training $verb at step $step, but the final checkpoint could not " +
                    "be written. Free some storage and train again so the model can be tested."
                stopped -> "Training stopped at step $step. Checkpoint saved; open the " +
                    "playground to test it."
                else -> "Training finished at step $step. Best measured loss " +
                    String.format("%.4f", bestLoss) + "."
            }
            finish(modelId, status, message, state.copy(step = step, totalSteps = totalSteps, bestLoss = bestLoss))
        } catch (cancel: CancellationException) {
            // Another run is taking over: save what we have and let the
            // cancellation propagate instead of reporting a training error.
            runCatching {
                val path = store.lastCheckpoint(modelId).absolutePath
                model?.takeIf { it.isOpen }?.saveCheckpoint(path, true)
            }
            throw cancel
        } catch (stopped: TrainingStopped) {
            // The user stopped during preparation, before the step loop ran.
            val last = currentState ?: TrainingLiveState(modelId, TrainingStatus.STOPPED)
            finish(
                modelId,
                TrainingStatus.STOPPED,
                "Training stopped before the first step. No checkpoint was changed.",
                last
            )
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
            stopTarget = null
            runCatching { model?.close() }
            releaseWakeLock()
        }
    }

    private suspend fun loadOrTrainTokenizer(
        manifest: ModelManifest,
        corpus: String,
        corpusCap: Int
    ): BpeTokenizer {
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
                maxTrainingChars = min(corpus.length, corpusCap)
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

    private fun openModel(manifest: ModelManifest, run: RunControl): NativeModel {
        val cfg = manifest.config
        val last = store.lastCheckpoint(manifest.id)
        val recovery = store.recoveryCheckpoint(manifest.id)
        val best = store.bestCheckpoint(manifest.id)
        // Try the newest readable checkpoint first: weights-only recovery may
        // be ahead of last, and a corrupt file must not strand the user.
        val candidates = listOf(recovery, last, best)
            .filter { CheckpointHeader.hasMagic(it) }
            .mapNotNull { file -> CheckpointHeader.readStep(file)?.let { file to it } }
            .sortedByDescending { it.second }

        var model: NativeModel? = null
        var selected: File? = null
        for ((file, _) in candidates) {
            val loaded = runCatching { NativeModel.load(file.absolutePath, withOptimizer = true) }
                .getOrNull() ?: continue
            val c = loaded.config
            val matches = c.vocabSize == cfg.vocabSize && c.contextLength == cfg.contextLength &&
                c.dModel == cfg.dModel && c.layers == cfg.layers && c.heads == cfg.heads &&
                c.dFf == cfg.dFf && c.tiedEmbeddings == cfg.tiedEmbeddings
            if (matches) {
                model = loaded
                selected = file
                break
            }
            loaded.close()
        }

        run.resumedWithRecovery = selected != null && selected != last

        if (model == null) {
            // A forked series version has no checkpoint of its own yet, but it
            // carries the previous version's weights. Start a genuinely new run:
            // step 0, fresh AdamW moments, same architecture and tokenizer.
            val init = store.initCheckpoint(manifest.id)
            if (CheckpointHeader.hasMagic(init)) {
                val seeded = runCatching {
                    NativeModel.load(init.absolutePath, withOptimizer = false)
                }.getOrNull()
                val matches = seeded?.config?.let { c ->
                    c.vocabSize == cfg.vocabSize && c.contextLength == cfg.contextLength &&
                        c.dModel == cfg.dModel && c.layers == cfg.layers && c.heads == cfg.heads &&
                        c.dFf == cfg.dFf && c.tiedEmbeddings == cfg.tiedEmbeddings
                } == true
                if (seeded != null && matches) {
                    seeded.resetStep()
                    model = seeded
                } else {
                    seeded?.close()
                }
            }
        }
        if (model == null) {
            model = NativeModel.create(cfg, System.nanoTime())
                ?: throw TrainingFailure(
                    "The model could not be allocated.",
                    "Close other apps or create a smaller model."
                )
        }
        if (cfg.bits in 2..8) {
            model.setQuantization(cfg.bits, 32)
        }
        if (!last.exists() || selected == null) {
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
        runCatching {
            val manifest = store.loadManifest(modelId)
            if (manifest != null) {
                store.modelCardFile(modelId).writeText(ModelCard.build(manifest, finalState))
            }
        }
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
        val type = when {
            Build.VERSION.SDK_INT >= 34 -> ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ->
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            else -> 0
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

    override fun onTaskRemoved(rootIntent: Intent?) {
        control.checkpointRequested = true
        if (store.activeModelId() != null && currentModelId != null && !control.stopRequested) {
            TrainingRestartReceiver.schedule(this, 5_000L)
        }
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        control.checkpointRequested = true
        val restart = store.activeModelId() != null && !control.stopRequested
        control.stopRequested = true
        // Unblock a step that is currently executing so the job can cancel and
        // save promptly instead of running to completion.
        stopTarget?.requestStop()
        // Stop accepting commands, then tear the run down. The job's own
        // cancellation handler saves a checkpoint.
        commands.close()
        commandJob.cancel()
        job?.cancel()
        releaseWakeLock()
        scope.cancel()
        if (restart) {
            TrainingRestartReceiver.schedule(this, 10_000L)
        }
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

/** Signals a cooperative stop requested by the user, not a failure. */
private class TrainingStopped : Exception()
