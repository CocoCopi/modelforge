package com.modelforge.app.data.hf

import com.jakewharton.retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Response
import retrofit2.Retrofit
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query
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

data class HfApiBundle(
    val publicApi: HfApi,
    val serverApi: HfDatasetServerApi
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
        val client = OkHttpClient.Builder()
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
            serverApi = serverRetrofit.create(HfDatasetServerApi::class.java)
        )
    }
}
