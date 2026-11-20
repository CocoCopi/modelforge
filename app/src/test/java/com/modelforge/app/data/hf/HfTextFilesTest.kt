package com.modelforge.app.data.hf

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream

class HfTextFilesTest {

    private class FakeSource(
        private val listings: Map<String, List<HfTreeEntry>>,
        private val contents: Map<String, String>
    ) : HfFileSource {
        var opens = 0
            private set

        override suspend fun list(
            datasetId: String,
            path: String,
            revision: String
        ): List<HfTreeEntry> = listings[path] ?: emptyList()

        override suspend fun open(
            datasetId: String,
            path: String,
            maxBytes: Long,
            revision: String
        ): InputStream {
            opens++
            val text = contents[path] ?: error("no content for $path")
            val bytes = text.toByteArray(Charsets.UTF_8)
            return ByteArrayInputStream(bytes.copyOf(minOf(bytes.size.toLong(), maxBytes).toInt()))
        }
    }

    private fun file(path: String, size: Long = 10_000L) = HfTreeEntry("file", path, size)

    private fun directory(path: String) = HfTreeEntry("directory", path, 0L)

    @Before
    fun resetLimiter() = HfRateLimiter.reset()

    @Test
    fun recognises_data_files_but_not_metadata() {
        assertTrue(HfTextFiles.isTextPath("data/train.jsonl"))
        assertTrue(HfTextFiles.isTextPath("stories.txt.gz"))
        assertTrue(HfTextFiles.isTextPath("pairs.csv"))
        assertTrue(!HfTextFiles.isTextPath("README.md"))
        assertTrue(!HfTextFiles.isTextPath("dataset_infos.json"))
        assertTrue(!HfTextFiles.isTextPath("config.yaml"))
        assertTrue(!HfTextFiles.isTextPath("weights.parquet"))
    }

    @Test
    fun keeps_only_the_requested_split() {
        val entries = listOf(
            directory("data"),
            file("data/train-00000-of-00002.jsonl"),
            file("data/train-00001-of-00002.jsonl"),
            file("data/test-00000-of-00001.jsonl")
        )
        val train = HfTextFiles.candidates(entries, "train")
        assertEquals(2, train.size)
        assertTrue(train.all { it.path.contains("train") })
        val test = HfTextFiles.candidates(entries, "test")
        assertEquals(1, test.size)
        assertTrue(test.single().path.contains("test"))
    }

    @Test
    fun uses_a_single_unnamed_file_for_the_split() {
        val entries = listOf(file("databricks-dolly-15k.jsonl", 13_000_000))
        val chosen = HfTextFiles.candidates(entries, "train")
        assertEquals(1, chosen.size)
    }

    @Test
    fun prefers_jsonl_over_json() {
        val entries = listOf(
            file("data.json", 500_000),
            file("data.jsonl", 500_000)
        )
        assertEquals("data.jsonl", HfTextFiles.candidates(entries, "train").first().path)
    }

    @Test
    fun streams_jsonl_columns_in_order() {
        val rows = mutableListOf<String>()
        val input = """
            {"instruction":"a","response":"b"}
            {"instruction":"c","response":"d"}
        """.trimIndent().byteInputStream()
        HfTextFiles.forEachRow(input, "train.jsonl", listOf("instruction", "response")) {
            rows += it
            true
        }
        assertEquals(listOf("a\nb", "c\nd"), rows)
    }

    @Test
    fun streams_objects_out_of_a_json_array() {
        val rows = mutableListOf<String>()
        val input = """[{"text":"one"},{"text":"two"},{"text":"three"}]""".byteInputStream()
        HfTextFiles.forEachRow(input, "data.json", listOf("text")) { row ->
            if (row == "three") {
                false
            } else {
                rows += row
                true
            }
        }
        assertEquals(listOf("one", "two"), rows)
    }

    @Test
    fun parses_csv_quotes_and_embedded_newlines() {
        val rows = mutableListOf<String>()
        val input = "text,label\n\"hello, world\",a\n\"line1\nline2\",b\n".byteInputStream()
        HfTextFiles.forEachRow(input, "train.csv", listOf("text")) { rows += it; true }
        assertEquals(listOf("hello, world", "line1\nline2"), rows)
    }

    @Test
    fun reads_plain_text_one_line_per_row() {
        val rows = mutableListOf<String>()
        val input = "first story\n\nsecond story\n".byteInputStream()
        HfTextFiles.forEachRow(input, "stories.txt", emptyList()) { rows += it; true }
        assertEquals(listOf("first story", "second story"), rows)
    }

    @Test
    fun joins_endoftext_documents_in_plain_text() {
        val rows = mutableListOf<String>()
        val input = "line one\nline two\n<|endoftext|>\nnext story\n".byteInputStream()
        HfTextFiles.forEachRow(input, "stories.txt", emptyList()) { rows += it; true }
        assertEquals(listOf("line one\nline two", "next story"), rows)
    }

    @Test
    fun downloads_directly_and_reports_stats() = runBlocking {
        val source = FakeSource(
            listings = mapOf(
                "" to listOf(file("train.jsonl", 500_000))
            ),
            contents = mapOf(
                "train.jsonl" to buildString {
                    repeat(5) { append("""{"text":"row $it"}""").append('\n') }
                }
            )
        )
        val dest = File.createTempFile("modelforge-direct-", ".txt")
        try {
            val stats = HfTextFiles.download(
                source = source,
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 3,
                maxBytes = 1_000_000,
                destination = dest
            )
            assertNotNull(stats)
            assertEquals(3L, stats!!.rows)
            assertEquals(3, dest.readLines().size)
            assertEquals(listOf("row 0", "row 1", "row 2"), dest.readLines())
            assertEquals(1, source.opens)
        } finally {
            dest.delete()
        }
    }

    @Test
    fun falls_back_when_the_file_has_no_requested_column() = runBlocking {
        val source = FakeSource(
            listings = mapOf("" to listOf(file("train.jsonl"))),
            contents = mapOf("train.jsonl" to """{"other":"x"}""" + "\n")
        )
        val dest = File.createTempFile("modelforge-direct-", ".txt")
        try {
            val stats = HfTextFiles.download(
                source = source,
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 10,
                maxBytes = 1_000_000,
                destination = dest
            )
            assertNull(stats)
            assertTrue(!dest.exists() || dest.length() == 0L)
        } finally {
            dest.delete()
        }
    }

    @Test
    fun looks_one_level_down_when_the_root_has_no_data_files() = runBlocking {
        val source = FakeSource(
            listings = mapOf(
                "" to listOf(directory("data")),
                "data" to listOf(file("data/train-00000-of-00001.jsonl"))
            ),
            contents = mapOf("data/train-00000-of-00001.jsonl" to """{"text":"deep"}""" + "\n")
        )
        val dest = File.createTempFile("modelforge-direct-", ".txt")
        try {
            val stats = HfTextFiles.download(
                source = source,
                datasetId = "owner/data",
                config = "default",
                split = "train",
                columns = listOf("text"),
                maxRows = 10,
                maxBytes = 1_000_000,
                destination = dest
            )
            assertNotNull(stats)
            assertEquals("deep", dest.readText().trim())
        } finally {
            dest.delete()
        }
    }

    @After
    fun clearLimiter() = HfRateLimiter.reset()
}
