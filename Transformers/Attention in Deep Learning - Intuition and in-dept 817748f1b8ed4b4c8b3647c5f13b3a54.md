# Attention in Deep Learning - Intuition and in-depth analysis

## A quick overview of the Encoder-Decoder architecture

In this section we will be taking a high level view on the popular encoder-decoder architecture and get a basic idea on how it works. What has this got to do with Attention - the limitations of this architecture inspired the attention mech and we will understand why these limitations occur

The encoder decoder architecture has 5 main parts that we want to talk about

### 1. The input sequence

For purposes of this task let us consider we are translating English to French. Lets assume that the sentence we want to translate is "Economic growth has slowed down in recent years." Then the input sequence is basically the words of the sentence. Now NNs can't process raw text so we need to encode them in some numeric format. Of course we can use one hot encoding for this purpose but a far better approach is to use word embeddings that can capture the similarities between the words.

So the input sequence is basically a set of vector representations for each word 

### 2. Encoder stack of NNs

A common misconception is that encoder decoder architectures always use rnns for encoding.but we can use an nn which can take in a set of inputs and compute a set of features from it. And since all NNs are feature extrators we can use any nn for this. We will see later in the post a use of CNN as the encoder to extract features from images. Here we will stick to RNNs as it will allow us to process sequential input. 

For simplicity consider that the encoder comprises a single RNN block which processes each input token one by one

```bash
new_hidden_state = RNN(input_token_at_that_time_step, prev_hidden_state)
h_T = RNN(i_T, h_{T-1})
```

//// enter animation here

### 3. Hidden state

Thus the final op of this encoder block is h_T of **any specified dimension** that basically summarizes all important information from the input sequence

Explain figure: [https://arxiv.org/pdf/1409.3215.pdf](https://arxiv.org/pdf/1409.3215.pdf)

### 4. Decoder stack of NNs

Now that we have a hidden state that has learned the important aspects (features) of the input sentence, we need a mechanism that can process these sequence of features and produce the outputs.

Again its not necessary to use RNNs but since we are dealing with sequential data, for this task it makes sense to use a stack of RNNs as the decoder module. Again for simplicity we consider that we have only one RNN as the decoder

The RNN decoder unit takes as inputs: 

- previous output word embedding
- previous hidden state
- the hidden state h_T - the encoder output

Using these inputs

/// simple block diagram

```bash
new_hidden_sate = RNN_decoder(prev_op_word_vector, prev_hidden_state, h_T)
```

Let us understand what information each of these inputs provides to the decoder

- previous output word embedding - given that the prev word generated was xyz what should be the next word?
- previous hidden state - as each new_hidden_sate, is computed using prev_hidden_state, this input captures info about all the outputs which have been generated so far. So given I have seen [token1, token2, tokken3] as op what should be the next token
- h_T - this gives info about the entire input sequence

So, to summarize at each step the decoder mainly tries to compute "given that I have generated the op [token1, token2, tokken3,..] and given that the prev word I generated was [...] and given all the input features, what should be the next hidden state?"

How to go from a hidden state to an op word?

Again, as we had the english words converted to word vectors we have the French word vectors as well. Given a new hidden state from the op decoder we compute the dot product with each of the op vectors. 

/// diag

The dot product bw 2 vectors is a scalar that can be interpreted as high when the vectors are closely aligned (the features are similar) and small when the vectors are dissimilar

So we get a set of N scores if there are N possible op French words

Instead of picking out the word corresponding to the max score we first run a softmax on the scores to bring them to the range 0→1 so that we can interpret them as probabilities. The inituitive reason for doing this is so that we can interpret the results for e.g: "the translation of x is x2 with 85% probability"

### 5. Output sequence