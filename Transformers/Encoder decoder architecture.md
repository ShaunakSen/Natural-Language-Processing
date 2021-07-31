# Encoder decoder architecture

Notes on the blog post by Kyunghyun Cho: https://developer.nvidia.com/blog/introduction-neural-machine-translation-with-gpus/

## Background and motivation of using RNNs

![https://i.imgur.com/rwhDeMX.png](https://i.imgur.com/rwhDeMX.png)

![https://i.imgur.com/V8p48fE.png](https://i.imgur.com/V8p48fE.png)

## Encoder-Decoder Architecture for Machine Translation

I’m not a neuroscientist or a cognitive scientist, so I can’t speak authoritatively about how the brain works. However, if I were to guess what happens in my brain when I try to translate a short sentence in English to Korean, my brain encodes the English sentence into a set of neuronal activations as I hear them, and from those activations, I decode the corresponding Korean sentence. In other words, the process of (human) translation involves the encoder which turns a sequence of words into a set of neuronal activations (or spikes, or whatever’s going on inside a biological brain) and the decoder which generates a sequence of words in another language, from the set of activations

This idea of encoder-decoder architectures is the basic principle behind neural machine translation. In fact, this type of architecture is at the core of deep learning, where the biggest emphasis is on **learning a good representation**. In some sense, you can always cut any neural network in half, and call the first half an encoder and the other a decoder.

### The Encoder

We start from the encoder, a straightforward application of a recurrent neural network, based on its property of sequence summarization. If you recall the previous post, this should be very natural. In short, we apply the recurrent activation function recursively over the input sequence, or sentence, until the end when the final internal state of the RNN h_T is the summary of the whole input sentence.

First, each word in the source sentence is represented as a so-called one-hot vector, or 1-of-K coded vector as in Figure 3. This kind of representation is the dumbest representation you can ever find. Every word is equidistant from every other word, meaning that it does not preserve any relationships among them.

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure-3_one-hot.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure-3_one-hot.png)

Figure 3. Step 1: A word to a one-hot vector.

We take a hierarchical approach to extracting a sentence representation, a vector that summarizes the input sentence. In that hierarchy, the first step is to obtain a meaningful representation of each word. But, what do I mean by “meaningful” representation? A short answer is “we let the model learn from data!”, and there isn’t any longer answer.

The encoder linearly projects the 1-of-K coded vector w_i (see Figure 3) with a matrix E which has as many columns as there are words in the source vocabulary and as many rows as you want (typically, 100 – 500.) This projection s_i = E.w_i, shown in Figure 4, results in a continuous vector for each source word, and each element of the vector is later updated to maximize the translation performance. I’ll get back to what this means shortly.

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure4_continuous-space.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure4_continuous-space.png)

Figure 4. Step 2: A one-hot vector to a continuous-space representation.

At this point, we have transformed a sequence of words into a sequence of continuous vectors s_is, and the recurrent neural network comes in. At the end of the last post, I said that one of the two key capabilities of the RNN was a capability of summarizing a sequence, and here, I will use an RNN to summarize the sequence of continuous vectors corresponding to the words in a source sentence. Figure 5 illustrates how an RNN does it.

I can write this process of summarization in mathematical notation as

[https://s0.wp.com/latex.php?latex=h_i+%3D+%5Cphi_%7B%5Ctheta%7D%28h_%7Bi-1%7D%2C+s_i%29&bg=ffffff&fg=000&s=0&c=20201002](https://s0.wp.com/latex.php?latex=h_i+%3D+%5Cphi_%7B%5Ctheta%7D%28h_%7Bi-1%7D%2C+s_i%29&bg=ffffff&fg=000&s=0&c=20201002)

where h_0 is an all-zero vector. In other words, after the last word’s continuous vector s_T is read, the RNN’s internal state h_T represents a summary of the whole source sentence (the solid red circle represents h_T).

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure5_summarization.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure5_summarization.png)

Figure 5. Step 3: Sequence summarization by a recurrent neural network.

Now that we have a summary vector, a natural question comes to mind: “what does this summary vector look like?” I would love to spend hours talking about what that summary vector should look like, what it means and how it’s probably related to representation learning and deep learning, but I think one figure from [Sutskever et al., 2014] says it all in a much more compact form (Figure 6).

From this, we can get a rough sense of the relative locations of the summary vectors in the original space. What we can see from Figure 6 is that the summary vectors do preserve the underlying structure, including semantics and syntax (if there’s a such thing as syntax); in other words, similar sentences are close together in summary vector space.

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure6_summary_vector_space.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure6_summary_vector_space.png)

Figure 6. 2-D Visualization of Sentence Representations from [Sutskever et al., 2014]. Similar sentences are close together in summary-vector space.

### The Decoder

Now that we have a nice fixed-size representation of a source sentence, let’s build a decoder, again using a recurrent neural network (the top half in Figure 2). Again, I will go through each step of the decoder. It may help to keep in mind that the decoder is essentially the encoder flipped upside down.

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure7_internal-hidden-state.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure7_internal-hidden-state.png)

Figure 7. Decoder – Step 1: Computing the internal hidden state of the decoder.

Let’s start by computing the RNN’s internal state z_i based on the summary vector h_T of the source sentence, the previous word u_{i-1} and the previous internal state z_{i-1}. Don’t worry, I’ll shortly tell you how to get the word. The new internal state z_i is computed by

[https://s0.wp.com/latex.php?latex=z_i+%3D+%5Cphi_%7B%5Ctheta%27%7D+%28h_T%2C+u_%7Bi-1%7D%2C+z_%7Bi-1%7D%29.&bg=ffffff&fg=000&s=0&c=20201002](https://s0.wp.com/latex.php?latex=z_i+%3D+%5Cphi_%7B%5Ctheta%27%7D+%28h_T%2C+u_%7Bi-1%7D%2C+z_%7Bi-1%7D%29.&bg=ffffff&fg=000&s=0&c=20201002)

z_i is a vector that we obtain for each op word

With the decoder’s internal hidden state z_i ready, we can now **score each target word based on how likely it is to follow all the preceding translated words given the source sentence.**

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure8_scoring.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure8_scoring.png)

Figure 8. Decoder – Step 2: Next word probability.

[https://s0.wp.com/latex.php?latex=e%28k%29+%3D+w_k%5E%7B%5Ctop%7D+z_i+%2B+b_k%2C&bg=ffffff&fg=000&s=0&c=20201002](https://s0.wp.com/latex.php?latex=e%28k%29+%3D+w_k%5E%7B%5Ctop%7D+z_i+%2B+b_k%2C&bg=ffffff&fg=000&s=0&c=20201002)

where w_k and b_k are the (target) word vector and a bias, respectively.

This is done for each k → 1 till no of words

Let’s forget about the bias b_k for now, and think of the first term, the dot product between two vectors. The dot product is larger when the target word vector w_k and the decoder’s internal state z_i are similar to each other, and smaller otherwise. Remember: a dot product gives the length of the projection of one vector onto another; if they are similar vectors (nearly parallel) the projection is longer than if they very different (nearly perpendicular). So this mechanism scores a word high if it aligns well with the decoder’s internal state.

Once we compute the score of every word, we now need to turn the scores into proper probabilities using

[https://s0.wp.com/latex.php?latex=p%28w_i%3Dk%7Cw_1%2C+w_2%2C+%5Cldots%2C+w_%7Bi-1%7D%2C+h_T%29+%3D+%5Cfrac%7B%5Cexp%28e%28k%29%29%7D%7B%5Csum_%7Bj%7D+%5Cexp%28e%28j%29%29%7D.&bg=ffffff&fg=000&s=0&c=20201002](https://s0.wp.com/latex.php?latex=p%28w_i%3Dk%7Cw_1%2C+w_2%2C+%5Cldots%2C+w_%7Bi-1%7D%2C+h_T%29+%3D+%5Cfrac%7B%5Cexp%28e%28k%29%29%7D%7B%5Csum_%7Bj%7D+%5Cexp%28e%28j%29%29%7D.&bg=ffffff&fg=000&s=0&c=20201002)

This type of normalization is called softmax

Now we have a probability distribution over the target words, which we can use to select a word by sampling the distributionas Figure 9 shows. After choosing the i-th word, we go back to the first step of computing the decoder’s internal hidden state (Figure 7), scoring and normalizing the target words (Figure 8) and selecting the next (i+1)-th word (Figure 9), repeating until we select the end-of-sentence word (<eos>).

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure9_sampling.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/06/Figure9_sampling.png)

Figure 9. Decoder – Step 3: Sampling the next word.

## The Trouble with Simple Encoder-Decoder Architectures

**In the encoder-decoder architecture, the encoder compresses the input sequence as a fixed-size vector from which the decoder needs to generate a full translation. In other words, the fixed-size vector, which I’ll call a context vector, must contain every single detail of the source sentence. Intuitively, this means that the true function approximated by the encoder has to be extremely nonlinear and complicated. Furthermore, the dimensionality of the context vector must be large enough that a sentence of any length can be compressed.**

In my paper “On the Properties of Neural Machine Translation: Encoder-Decoder Approaches” presented at SSST-8, my coauthors and I empirically confirmed that translation quality dramatically degrades as the length of the source sentence increases when the encoder-decoder model size is small. Together with a much better result from Sutskever et al. (2014), using the same type of encoder-decoder architecture, this suggests that the representational power of the encoder needed to be large, which often means that the model must be large, in order to cope with long sentences (see Figure 1).

![https://developer-blogs.nvidia.com/wp-content/uploads/2015/07/Figure1_BLEUscore_vs_sentencelength-624x459.png](https://developer-blogs.nvidia.com/wp-content/uploads/2015/07/Figure1_BLEUscore_vs_sentencelength-624x459.png)

Due to its sequential nature, a recurrent neural network tends to remember recent symbols better. In other words, the further away an input symbol is from j, the less likely the RNN’s hidden state, remembers it perfectly.