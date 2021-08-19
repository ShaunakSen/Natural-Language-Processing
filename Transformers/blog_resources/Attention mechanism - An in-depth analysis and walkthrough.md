# Attention mechanism - An in-depth analysis and walkthrough - Part 1

//// introduction - explain motivation of studying attention + how we are going to cover it in each part

Often concepts in deep learning are not quite interpretable, but we will see this is not the case with the Attention mechanism and we will logically build towards the full architecture

## A quick overview of the Encoder-Decoder architecture

/// add reference

In this section we will be taking a high level view on the popular encoder-decoder architecture and get a basic idea on how it works. What has this got to do with Attention - the limitations of this architecture inspired the attention mech and we will understand why these limitations occur. 

The encoder decoder architecture has 5 main parts that we want to talk about. These 5 parts are highlighted in the diagram below and we will understand them one by one

![seq2seq_basic.svg](https://i.imgur.com/PoXLyKb.png)

### 1. The input sequence

Let us consider we are translating English to French. Lets assume that the sentence we want to translate is "Economic growth has slowed down in recent years." Then the input sequence is basically the words of the sentence. 

Now neural networks can't process raw text so we need to encode them in some numeric format. So each word should become a vector of numbers of a predefined dimension that we can input to the network. *Why a vector, and not a scalar - each word has multiple characteristics and vector allows for that info to be captured*. Of course we can use one-hot encoding for this purpose but a far better approach is to use word embeddings that can capture the similarities between the words. To get an idea on how word embeddings work and why we should use them refer to []

To explain things very simply, we have a big embedding matrix with no of rows == no of words in vocabulary and no of cols/dimensions = no of dimensions we want each word to have (typically 64/128/512 - here we consider 4 for simplicity). We lookup each word in the input sequence in this matrix and map it to a corresponding vector. This allows us to express each word as a vector and the way these vectors are defined is that words similar to each other are mapped closer to each other. 

So the input sequence is basically a set of vector representations for each word 

[word_embeddings.mp4](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/word_embeddings.mp4)

We can see in the animation above how each input token gets mapped to a corresponding vector. From now on, whenever you see me saying things like "the model receives a word as input", I mean that it takes in the learned vector corresponding to the word

### 2. Encoder stack of NNs

A common misconception is that encoder decoder architectures always use RNNs for encoding.  But we can use any neural network which can take in a set of inputs and compute a set of features from it. And since all NNs are feature extractors we can use any neural network for this. We will see later in the post a use of CNN as the encoder to extract features from images. Here we will stick to RNNs as it will allow us to process sequential input. 

For simplicity consider that the encoder comprises a single RNN block which processes each input token one by one. The working of this encoder RNN can be best understood with the help of an animation:

[encoder.mp4](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/encoder.mp4)

TODO: set initial values of hidden state

1. At each time step (t) the encoder RNN receives the current input $x_t$ and the previous hidden state $h_{t-1}$. It then computes the next hidden state $h_t$ (this hidden state will again be fed back to the network at time step t+1). In addition to the hidden step, the RNN also produces an output vector, which is not important here

    $$h_t = \texttt{enocder\_RNN}(x_t, h_{t-1})$$

2. The initial hidden state $h_0$ is assumed to be all 0s
3. $h_8$ is the final hidden state

### 3. Hidden state

Thus the final op of this encoder block is $h_8$ of **any specified dimension** that basically summarizes all important information from the input sequence. This is because to create $h_8$ we need to compute $h_7$, which in turn requires $h_6$ and so on.. So the final hidden state vector can be refered to as the "context" vector as, for a well trained network this vector summarizes all useful information (features) from the input sentence that can be used by the decoder. 

*Notice how we loosely refer to this context vector as a set of features of the input text, so essentially the encoder network acts as a **feature extractor** for the input.*

![https://i.imgur.com/SM9pprY.png](https://i.imgur.com/SM9pprY.png)

[https://arxiv.org/pdf/1409.3215.pdf](https://arxiv.org/pdf/1409.3215.pdf)

In the above diagram we plot the hidden state vectors for a number of input sentences. The authors in this paper have used an LSTM network and used PCA to reduce the hidden state dimensions to 2 for plotting on the 2D co-ordinate system. Notice how the hidden state seems to have efficiently encoded the input sentences as sentences of similar information are clustered together. Also note that the input sentences are fairly short in these results

### 4. Decoder stack of NNs

Now that we have a hidden state that has learned the important aspects (features) of the input sentence, we need a mechanism that can process these sequence of features and produce the outputs.

Again its not necessary to use RNNs but since we are dealing with sequential data, for this task it makes sense to use a stack of RNNs as the decoder module. For simplicity we consider that we have only one RNN as the decoder

The RNN decoder, at each time step t takes as inputs: 

- previous output word embedding $op_{t-1}$
- previous hidden state $s_{t-1}$
- the hidden state $h_8$ - the encoder output

Using these inputs the decoder basically computes 

- The next hidden state - $s_t$

The RNN decoder process can be summarized as:

$$s_t = \texttt{decoder\_RNN}(op_{t-1}, s_{t-1}, h_8)$$

[https://www.youtube.com/watch?v=NIyiGIv16j8](https://www.youtube.com/watch?v=NIyiGIv16j8)

There is a lot of things going on in this animation so let us go through it:

1. At the start we have the encoding process which takes in the word embeddings as shown in [] and computes a context vector $h_8$
2.  At each time step the decoder computes the hidden state $s_t$ using the equation $s_t = \texttt{decoder\_RNN}(op_{t-1}, s_{t-1}, h_8)$ as shown in the diagram
3. Using this hidden state we map to an output word $op_t$ which is the corresponding translated word in French - we will see what goes on in this "Mapping process to output" block soon.
4. Note that initially $s_0$ is set to a vector of 0s and $op_0$ is simply the embedding for the <START> token

Let us understand what information each of the inputs provides to the decoder

- previous output word embedding - given that the previous word generated was $op_{t-1}$ what should be the next word?
- previous hidden state - as each new hidden state $s_t$, is computed using $s_{t-1}$ which is calculated using $s_{t-2}$ and so on... So $s_t$ input captures information about all the outputs which have been generated so far, i.e.:  given we have generated $[op_1, op_2, ... op_k]$ what should be the next token to be generated
- $h_8$ - this gives information about the entire input sequence - this vector has summarized the input text and serves as a kind of "context" vector

So, to summarize at each step the decoder mainly tries to compute "given that I have generated the op  $[op_1, op_2, ... op_k]$ and given that the prev word I generated was $op_{t-1}$ and given all the input features, what should be the next hidden state?"

### 5. Mapping of hidden state → output word

We have so far seen how we use the decoder network to compute a vector $s_t$ at each time step which is supposed to encode information about an output word - but how can we map this vector to an output word?

As we saw in [diag] we had a set of embedding vectors for each English word in form of a matrix from which we looked up the required input vectors

Similarly we have the corresponding French word embeddings stored in a matrix. Once we get the hidden state $s_t$ we can compute the "similarity" between $s_t$ and each French word embedding. We pick the word which gives the highest similarity score as $op_t$

### Dot product as a similarity measure

To explain this let me borrow from the [[https://mlvu.github.io/](https://mlvu.github.io/)]

![dot_product.svg](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/dot_product.svg)

Imagine we have 2 vectors - for a movie $m$ and for a user $u$

Each dimension of these vectors is like a feature for the movie and user respectively and we can see that the movie has action in it and the user also likes action a lot

The dot product between these two vectors gives us a $score$ which is a scalar and its high in this case due to the fact that the user likes action and the movie has action - so the dot product returns a high value of $score$ due to the large contribution of the term $u_2.m_2$ when the 2 vectors have similar features.

Similarly we can imagine that if the user hates romance and the movie has romance then $u_1$ is a large negative, $m_1$ is a large positive and the score is low due to the negative contribution of the term $u_1.m_1$

In terms of vector spaces, if we plot $m$ and $u$, then if these vectors are closely aligned the dot product score is more, i.e. if the angle of separation ($\theta$) is less then $cos(\theta)$ will be more (for $\theta < 90 \degree$: $cos(\theta)$ is monotonically decreasing function)

So, to conclude the dot product is a very intuitive and easy to compute metric for similarity between 2 vectors

Coming back to the original problem, we compute the dot product between between $s_t$ and each French word embedding - this gives us a set of similarity scores scores. These scores can range anywhere between $(-\infty, +\infty)$, so we apply a softmax (add ref) to scale it to a range of $[0,1]$ so that we can interpret these scores as probabilities. The intuitive reason for doing this is so that we can interpret the results for e.g.: "the translation of 'growth' from English to French is 'croissance' with 85% probability".

In the animation below I have depicted this process for the mapping from $s_2 \rightarrow o_2$

All outputs are similarly mapped from the decoder hidden state to a French translation

[output_mapping.mp4](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/output_mapping.mp4)

## Limitations of this architecture

We have developed this architecture in light of the task of converting English Sentences to French and it seems to be a great solution, and it really is! In fact, ENc -dec architectures are extensively used by Google Translate and we all know how great it is! ([https://research.google/pubs/pub46151/](https://research.google/pubs/pub46151/))

However if we consider tasks like text summarization or question-answering systems like chatbots which have to process and remember information from a large piece of text in order to process outputs, the limitations of this architecture become apparent. The main problem is with h_T which is a vector of fixed dimensionality that is supposed to somehow encode all relevant info from the input sequence into this latent space. For translating short pieces of text this is acceptable but for encoding really large input texts, this method fails. Think of Google assistant or Siri from a few years back. It was great at understanding simple queries and answering them but it could not carry out a long conversation as we did not have way of preserving long-term dependencies yet. Of course we can scale up the dimension of this context vector to preserve more information but that will increase the training time and since we are using a sequential nw like RNNs we cant even parallelize it as at each time step we need as input the op of prev time step. So simply increasing the dimensionality is not a feasible solution.

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/07/Figure1_BLEUscore_vs_sentencelength-624x459.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/07/Figure1_BLEUscore_vs_sentencelength-624x459.png)

Drop in performance of seq→seq architectures with increased sentence length Credits: [https://developer.nvidia.com/blog/introduction-neural-machine-translation-gpus-part-3/](https://developer.nvidia.com/blog/introduction-neural-machine-translation-gpus-part-3/)

The BLEU score is simply a metric that compares how good an output text from the model is close to the true reference text. It does so by matching N-grams and the higher the score the better. You can read more on this here: [https://machinelearningmastery.com/calculate-bleu-score-for-text-python/](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)

## Building towards Attention

We have spent a lot of time understanding how sequence to sequence models work but this is a topic on Attention mechanism. The need for understanding sequence to sequence models is because attention was first introduced to solve the main limitation of this architecture and understanding this will help us get an intuitive sense of how it solves the problem.

So far we have understood how sequence to sequence architecture works and established that our main limitation is that a **single static** context vector is not enough to efficiently encode all input information of long sequences. 

*Well, the logical solution is to simply have multiple context vectors that change at each time step and feed them into the decoder at each time step.*

Also, Remember that in the sequence to sequence architecture, in the encoding process we had only used the last hidden state of the encoder $h_8$ in the decoding stage and discarded all previous hidden states.

[encoder.mp4](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/encoder.mp4)

TODO: set initial values of hidden state

The encoder computes hidden states $h_1, h_2, ..., h_8$. Instead of discarding all this information we can use these hidden states for the decoding process - remember again that we can loosely assume that these are features extracted from the input vectors. Also the purpose of the context vector is to somehow give us information on the input context.

Thus the context vector that will be passed to the decoder at time step t should be influenced by:

1. The previous hidden state of the decoder 
2. The hidden states of the encoder $h_1, h_2, ..., h_8$ - these are information about the input features

 

![https://i.imgur.com/QoAjXKb.png](https://i.imgur.com/QoAjXKb.png)

Again, if we think logically, consider we are generating the output at time step 2 i.e. $op_2$. Each of the input hidden states (extracted features from each input word) are not really equally important for generating $op_2$. Depending on the output we are generating, we need to assign a certain weight (attention) to each of the input features. 

*The weight that we assign to each input feature depending on the output we have generated so far is called **attention*** 

![https://i.imgur.com/HwPX8WB.png](https://i.imgur.com/HwPX8WB.png)

Here we have simply assigned weights to each hidden state/input feature. Note how equation 2 simply looks like a weighted average, where we assign different weights to the diff input features at each time step based on the outputs generated ($s_{t-1}$ encodes information of the previous outputs generated). Equation 2 simply tries to compute a context vector at each time step based on how  important each of the input features are necessary for predicting the word that comes after the hidden state $s_{t-1}$.

The next question is how to come up with these weights for each input feature at each time step? The authors of [] initially proposed to use a feed-forward neural network to figure this out. So this net receives an input feature vector and the previous hidden state vector and comes up with a scalar weight which is the attention weight to assign to that feature at that time step

 

![https://i.imgur.com/5iQeov8.png](https://i.imgur.com/5iQeov8.png)

In this way we calculate a set of attention weights for each time step t. Again, to normalize these weights and interpret them as probabilities, we pass these weights through a SoftMax operation. 

![https://i.imgur.com/vqk3Khp.png](https://i.imgur.com/vqk3Khp.png)

Note that since we are considering the previous hidden state $s_{t-1}$ while calculating each weight itself, we can simplify equation 2 as :

![https://i.imgur.com/a4txsHP.png](https://i.imgur.com/a4txsHP.png)

As mentioned before this is a weighted average over the input features where the weights are learned based on the i/p feature and the output generated so far. Thus we have solved the problem of the context vector being a single static vector which was not enough to summarize the input features efficiently, we now have a mechanism to compute the context vector dynamically at each time step