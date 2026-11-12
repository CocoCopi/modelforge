package com.modelforge.app.data.tokenizer

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class BpeTokenizerTest {

    private val sample = buildString {
        repeat(40) {
            append("One day, a little girl named Lily found a needle in her room. ")
            append("She wanted to share the needle with her mom. ")
            append("Hello, 世界! 🌍 café naïve.\n\n")
        }
    }

    @Test
    fun trainsRealMergesAndRoundTrips() {
        val tokenizer = BpeTokenizer.train(sample, targetVocab = 320, maxTrainingChars = 20_000)
        assertTrue("expected learned merges", tokenizer.vocabSize > 256)
        val ids = tokenizer.encode(sample)
        assertTrue("encoding should not be empty", ids.isNotEmpty())
        assertTrue("all ids in range", ids.all { it in 0 until tokenizer.vocabSize })
        assertEquals(sample, tokenizer.decode(ids))
        assertTrue(
            "BPE should compress repeated text",
            tokenizer.compressionRatio(sample) > 0.2
        )
    }

    @Test
    fun jsonRoundTripPreservesEncoding() {
        val tokenizer = BpeTokenizer.train(sample, targetVocab = 300, maxTrainingChars = 10_000)
        val restored = BpeTokenizer.fromJson(tokenizer.toJson())
        assertEquals(tokenizer.vocabSize, restored.vocabSize)
        assertArrayEquals(tokenizer.encode(sample), restored.encode(sample))
    }

    @Test
    fun trainingIsDeterministic() {
        val a = BpeTokenizer.train(sample, targetVocab = 300, maxTrainingChars = 10_000)
        val b = BpeTokenizer.train(sample, targetVocab = 300, maxTrainingChars = 10_000)
        assertEquals(a.toJson(), b.toJson())
    }

    @Test
    fun handlesEmptyAndShortInputs() {
        val tokenizer = BpeTokenizer.train(sample, targetVocab = 280, maxTrainingChars = 5_000)
        assertEquals("", tokenizer.decode(tokenizer.encode("")))
        val text = "a"
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)))
        val long = "space    space\t\tnewline\nend"
        assertEquals(long, tokenizer.decode(tokenizer.encode(long)))
    }
}
