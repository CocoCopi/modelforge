package com.modelforge.app.data.hf

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.Dispatcher
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Response
import okhttp3.ResponseBody
import retrofit2.Retrofit
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.Path
import retrofit2.http.Query
import retrofit2.http.Streaming
import retrofit2.http.Url
import java.util.concurrent.TimeUnit

interface HfApi {
    @GET("api/datasets")
    suspend fun searchDatasets(
        @Query("search") query: String,
        @Query("limit") limit: Int = 30
    ): List<HfDatasetSummary>

    @GET("api/datasets/{id}")
    suspend fun datasetDetail(
        @Path(value = "id", encoded = true) id: String
    ): HfDatasetDetail

    @GET("api/whoami-v2")
    suspend fun whoAmI(): HfWhoAmI
}

interface HfDatasetServerApi {
    @GET("info")
    suspend fun info(@Query("dataset") dataset: String): DatasetInfoResponse

    @GET("splits")
    suspend fun splits(@Query("dataset") dataset: String): SplitsResponse

    @GET("rows")
    suspend fun rows(
        @Query("dataset") dataset: String,
        @Query("config") config: String,
        @Query("split") split: String,
        @Query("offset") offset: Long,
        @Query("length") length: Int
    ): RowsResponse
}

/** Repository tree and raw file access, used to avoid the 100-row page API. */
interface HfFileApi {
    @GET("api/datasets/{id}/tree/{revision}")
    suspend fun treeRoot(
        @Path(value = "id", encoded = true) id: String,
        @Path(value = "revision", encoded = true) revision: String
    ): List<HfTreeEntry>

    @GET("api/datasets/{id}/tree/{revision}/{path}")
    suspend fun treeDir(
        @Path(value = "id", encoded = true) id: String,
        @Path(value = "revision", encoded = true) revision: String,
        @Path(value = "path", encoded = true) path: String
    ): List<HfTreeEntry>

    /**
     * A resolver URL, not a Hub API call: these have their own, much larger
     * budget (5,000 per 5 minutes for a signed-in user). A full URL is used
     * because repository paths may contain characters that a path parameter
     * would escape incorrectly.
     */
    @Streaming
    @GET
    suspend fun file(
        @Url url: String,
        @Header("Range") range: String? = null
    ): ResponseBody

    companion object {
        private const val ROOT = "https://huggingface.co/"

        fun fileUrl(datasetId: String, revision: String, path: String): String =
            ROOT.toHttpUrl().newBuilder()
                .addPathSegments("datasets")
                .addPathSegments(datasetId)
                .addPathSegments("resolve")
                .addPathSegments(revision)
                .addPathSegments(path)
                .build()
                .toString()
    }
}

data class HfApiBundle(
    val publicApi: HfApi,
    val serverApi: HfDatasetServerApi,
    val fileApi: HfFileApi? = null,
    /** Whether a Hugging Face token is configured for this client. */
    val authenticated: () -> Boolean = { false }
)

/** Adds the optional Hugging Face token without ever logging it. */
class AuthInterceptor(private val tokenProvider: () -> String?) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val token = tokenProvider()
        val request = if (!token.isNullOrBlank()) {
            chain.request().newBuilder()
                .header("Authorization", "Bearer $token")
                .build()
        } else {
            chain.request()
        }
        return chain.proceed(request)
    }
}

object HfClient {
    val json: Json = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
        isLenient = true
    }

    fun create(tokenProvider: () -> String? = { null }): HfApiBundle {
        // Dataset downloads keep many page requests in flight, so the OkHttp
        // dispatcher must be willing to run them concurrently instead of
        // silently queueing most of them behind the default per-host limit.
        val dispatcher = Dispatcher().apply {
            maxRequests = 64
            maxRequestsPerHost = 64
        }
        val client = OkHttpClient.Builder()
            .dispatcher(dispatcher)
            .addInterceptor(AuthInterceptor(tokenProvider))
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()

        val contentType = "application/json".toMediaType()
        val publicRetrofit = Retrofit.Builder()
            .baseUrl("https://huggingface.co/")
            .client(client)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()

        val serverRetrofit = Retrofit.Builder()
            .baseUrl("https://datasets-server.huggingface.co/")
            .client(client)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()

        return HfApiBundle(
            publicApi = publicRetrofit.create(HfApi::class.java),
            serverApi = serverRetrofit.create(HfDatasetServerApi::class.java),
            fileApi = publicRetrofit.create(HfFileApi::class.java),
            authenticated = { !tokenProvider().isNullOrBlank() }
        )
    }
}
