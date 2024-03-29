# Understanding Transformers

---

Seq2seq and basic attention: Andrew ng+ jay alamar + [https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)

[https://www.youtube.com/watch?v=zHY6WiTtGWQ](https://www.youtube.com/watch?v=zHY6WiTtGWQ)

Attention for transformer: Waterloo + jay alamar

Transformer: yannik + jay alamar

[https://www.youtube.com/watch?v=KmAISyVvE1Y](https://www.youtube.com/watch?v=KmAISyVvE1Y)

[https://www.youtube.com/watch?v=6l1fv0dSIg4](https://www.youtube.com/watch?v=6l1fv0dSIg4)

[https://www.youtube.com/watch?v=ah7_mfl7LD0](https://www.youtube.com/watch?v=ah7_mfl7LD0)

Positional embedding: [https://www.youtube.com/watch?v=dichIcUZfOw](https://www.youtube.com/watch?v=dichIcUZfOw)

watch this: [https://www.youtube.com/watch?v=qYcy6h1Rkgg](https://www.youtube.com/watch?v=qYcy6h1Rkgg)

---

Paper link: https://arxiv.org/pdf/1706.03762.pdf

Notes on the resources:

- [https://www.youtube.com/watch?v=iDulhoQ2pro](https://www.youtube.com/watch?v=iDulhoQ2pro)
- [https://www.youtube.com/watch?v=SMZQrJ_L1vo](https://www.youtube.com/watch?v=SMZQrJ_L1vo)
- [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

## Prerequisites - Seq - to - seq architectures

- [https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)

Sequence-to-sequence learning (Seq2Seq) is about training models to convert sequences from one domain (e.g. sentences in English) to sequences in another domain (e.g. the same sentences translated to French).

`"the cat sat on the mat" -> [Seq2Seq model] -> "le chat etait assis sur le tapis"`

This can be used for machine translation or for free-from question answering (generating a natural language answer given a natural language question) -- in general, it is applicable any time you need to generate text.

There are multiple ways to handle this task, either using RNNs or using 1D convnets. Here we will focus on RNNs.

**The trivial case: when input and output sequences have the same length**

When both input sequences and output sequences have the same length, you can implement such models simply with a Keras LSTM or GRU layer (or stack thereof). This is the case in this example script that shows how to teach a RNN to learn to add numbers, encoded as character strings:

![https://blog.keras.io/img/seq2seq/addition-rnn.png](https://blog.keras.io/img/seq2seq/addition-rnn.png)

One caveat of this approach is that it assumes that it is possible to generate target[...t] given input[...t]. That works in some cases (e.g. adding strings of digits) but does not work for most use cases. **In the general case, information about the entire input sequence is necessary in order to start generating the target sequence.**

**The general case: canonical sequence-to-sequence**

In the general case, input sequences and output sequences have different lengths (e.g. machine translation) and the entire input sequence is required in order to start predicting the target. This requires a more advanced setup, which is what people commonly refer to when mentioning "sequence to sequence models" with no further context. Here's how it works:

- A RNN layer (or stack thereof) acts as "encoder": it processes the input sequence and returns its own internal state. Note that we discard the outputs of the encoder RNN, only recovering the state. This state will serve as the "context", or "conditioning", of the decoder in the next step.
- Another RNN layer (or stack thereof) acts as "decoder": it is trained to predict the next characters of the target sequence, given previous characters of the target sequence. Specifically, it is trained to turn the target sequences into the same sequences but offset by one timestep in the future, a training process called "teacher forcing" in this context. Importantly, the encoder uses as initial state the state vectors from the encoder, which is how the decoder obtains information about what it is supposed to generate. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.

## Transformers - what is the use?

![http://jalammar.github.io/images/t/the_transformer_3.png](http://jalammar.github.io/images/t/the_transformer_3.png)

Lets take transformers to be a black box model for now, where we give an input and get an op

Here we are using it to convert French to English

Ok, but what is inside this giant black box?

- Encoders and Decoders

![http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png](http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)

The encoder-decoder combination actually has a bunch of stacked encoder and decoders

The choice of "6" encoders/decoders is a hyperparameter

![http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png](http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

## What exactly happens in each encoder?

Inside the encoder there are 2 things:

1. Self Attention
2. Feed forward NN

![http://jalammar.github.io/images/t/Transformer_encoder.png](http://jalammar.github.io/images/t/Transformer_encoder.png)

### What is self attention

Lets take an example of a single encoder-decoder pair and understand what is happening. 

![http://jalammar.github.io/images/t/Transformer_decoder.png](http://jalammar.github.io/images/t/Transformer_decoder.png)

The input that goes into the encoder is the vector

Each input word is converted into a vector of dimension 512 by the word2vec algorithm

![http://jalammar.github.io/images/t/embeddings.png](http://jalammar.github.io/images/t/embeddings.png)

512 is again a hyperparameter

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

![http://jalammar.github.io/images/t/encoder_with_tensors.png](http://jalammar.github.io/images/t/encoder_with_tensors.png)

Now that we have transformed the words into vectors, these are passed to the self-attention layer

Also these vectors are fed into the self-attention layer **in parallel**, unlike in RNNs where we feed one input after another

The ops of self-attention is `z1,z2,z3`, which are again vectors

These vectors then go into the Feed forward NN

The op of the Feed forward NN goes into encoder 2

### What is self-attention

Say the following sentence is an input sentence we want to translate:

`”The animal didn't cross the street because it was too tired”`

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

> Similarly for every word, it will be helpful if we can find out how strongly it is related to the other words in the sentence.

> So for each word, we need a score wrt every other word in the sentence

> As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

If you’re familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

![http://jalammar.github.io/images/t/transformer_self-attention_visualization.png](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

### Self attention in more details

Let us simplify this example considering just 2 words 

![http://jalammar.github.io/images/t/encoder_with_tensors_2.png](http://jalammar.github.io/images/t/encoder_with_tensors_2.png)

Let’s first look at how to calculate self-attention using vectors, then proceed to look at how it’s actually implemented – using matrices.

**Step 1**

The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a *Query vector, a Key vector, and a Value vector*. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

![http://jalammar.github.io/images/t/transformer_self_attention_vectors.png](http://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

X1 shape: 1x512. WQ shape: 512x64. q1 Shape: 1x64

```python
X1 x WQ = q1

X1 x WK = k1

X1 x WV = v1

X2 x WQ = q2

X2 x WK = k2

X2 x WV = v2
```

*So at the end of this step we have computed the q, k and v vectors for each input word in the sentence*

**Step 2**

The second step in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. ***We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.***

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

![http://jalammar.github.io/images/t/transformer_self_attention_score.png](http://jalammar.github.io/images/t/transformer_self_attention_score.png)

The word we are focusing on is like the "main query term". We multiply this query term with the keys of each of the other words. By doing so we kind of involve every other word in encoding this current word.  

**Step 3 and 4**

The third and forth steps are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

![http://jalammar.github.io/images/t/self-attention_softmax.png](http://jalammar.github.io/images/t/self-attention_softmax.png)

This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

**Step 5**

The fifth step is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

**Step 6**

The sixth step is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

![http://jalammar.github.io/images/t/self-attention-output.png](http://jalammar.github.io/images/t/self-attention-output.png)

z1 is the op for the first word, and it considers through the query-key lookup the importance of all other words in the sentence

Similarly z2 will be the op of the second word

That concludes the self-attention calculation. The resulting vector is z1, which we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. So let’s look at that now that we’ve seen the intuition of the calculation on the word level.

## Matrix Calculation of Self-Attention

The first step is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

Say we need to encode 2 words as in the fig

X = 2x512

WQ=WK=WV = 512 x 64

Q=K=V = 2x64

QxK^T = (2x64)x(64x2) = 2x2 

Q x K^T x V = (2x2) x (2x64) = 2x64 = dimension of op (Z)

So 2 rows of 64 dimensions representing the encoded ops of the 2 words in the sentence

![http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png](http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

## Multi-head Attention

The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

1. It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to.
2. It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

![http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png](http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)

Example showing 2 head attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices:

![http://jalammar.github.io/images/t/transformer_attention_heads_z.png](http://jalammar.github.io/images/t/transformer_attention_heads_z.png)

This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

![http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png](http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

Taking our previous example the op of the single-head attention was Z (2x64)

If we use 8-head attention Z = (2x 512) 64x8=512

W0 = 512 x 64

Z x W0 = (2x512) x (512x64) = 2x64 → this is then fed into the Feed forward NN

---

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

![http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

Now that we have touched upon attention heads, let’s revisit our example from before to see where the different attention heads are focusing as we encode the word “it” in our example sentence:

![http://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png](http://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png)

As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".

If we add all the attention heads to the picture, however, things can be harder to interpret:

![http://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png](http://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)

---

## CS480/680 Lecture 19: Attention and Transformer Networks

https://www.youtube.com/watch?v=OyFJWRnt_AY

### Attention in NLP:

In NLP, Language Modelling refers to the task of predicting the next word for a sequence or the task of recovering a missing word in a sequence

Most tasks in NLP can be formulated as a language modelling task, for e.g. in translation, we have a source sentence and maybe the initial few words of the translated sentence and we are looking to predict the next translated words. Same can be framed for sentiment analysis (how? : more examples on how to convert popular NLP tasks to Language Modelling)

Transformer Networks essentially use some Attention blocks and do away with RNNs and it has been shown to be very effective in LM tasks

![https://i.imgur.com/7kcqNvN.png](https://i.imgur.com/7kcqNvN.png)

///// TODO : complete this part

## Attention model Intuition

Video by AndrewNG: [https://www.youtube.com/watch?v=SysgYptB19](https://www.youtube.com/watch?v=SysgYptB198)

For encoder-decoder architectures of RNNs we have the following

ip → encoder → hidden state → decoder → op

But for long sequences we expect the encoder to take in the whole sentence at once and learn how to encode it into a hidden state effectively

This is not how a human would do this. They would probably read the sentence part by part and generate the op also part by part

This is why encoder-decoder architecture performance for long sentences reduce

![https://i.imgur.com/UHkLH2q.png](https://i.imgur.com/UHkLH2q.png)

This is the problem attention models aim to solve

Lets illustrate this with a short sentence, we want to translate this sentence from French to English

Say we use a bidirectional RNN for this purpose

Using the RNN for each of the words we have computed a rich set of features, and this is the op of the first RNN layer set

![https://i.imgur.com/dE2DdQf.png](https://i.imgur.com/dE2DdQf.png)

So the first set of RNN layer produces some feature and we will assume that the second set generates the corresponding English translations

a is the activation (hidden state) of the first layer of RNNs 

S is the activation (hidden state) of the second layer of RNNs 

![https://i.imgur.com/I53ANYi.png](https://i.imgur.com/I53ANYi.png)

Now the question is that when we want to generate the op word "Jane", which parts of the ip sentence should we look at - what the attention model computes is a set of **attention weights** that helps answer this

**`alpha_i,j : When we are trying to compute the ith word, how much attention should be give to the jth word`** 

**`C_1 : alpha_1,1 + alpha_1,2 + ... + alpha_1,N - This is the context for generating the first word`**

C_1 will be input to the first RNN unit of the second layer

Similarly for the second RNN unit, for generating the second word we will have a new hidden state S2 and a new set of attention wts (alpha_2,1, alpha_2,2 ... alpha_2,N)

Also since RNNs are sequential, the first generated word (here Jane) is also input to this second RNN

![https://i.imgur.com/TS4vMvv.png](https://i.imgur.com/TS4vMvv.png)

NOTE: here the context is labeled C, but it is different for each RNN block

Lets consider alpha_3,t i.e when we are trying to generate the 3rd word, the attention we are placing on t_th word will depend on the activation of the bidirectional RNN at time t and also on the state from the prev step (S_2)

In this way the RNN marches forward generating one word at a time until it generates <EOS> and at every step there is an attention wt (alpha_i,j) that tells it that when u are trying to generate the ith eng word how much attention should u pay to jth french word

## Mechanics of Attention model

Video by AndrewNG: https://www.youtube.com/watch?v=quoGRI-1l0A

In a bi-directional RNN, at every time step we have features computed from forward recurrence and backward recurrence, we will denote both as $<a^{\acute{t}}>$. So $<a^{\acute{t}}>$ will be the feature vector for time step t

![https://i.imgur.com/KQyx4KA.png](https://i.imgur.com/KQyx4KA.png)

Also, the next layer has:

![https://i.imgur.com/bLBGfDQ.png](https://i.imgur.com/bLBGfDQ.png)

The alpha_1,1 → alpha_1,N are basically the attention wts to compute the first op word

One requirement is that all these wts sum up to 1 (they should be normalized for varying input sentence length)

![https://i.imgur.com/w7sEiWJ.png](https://i.imgur.com/w7sEiWJ.png)

Just to simplify this, the equations imply:

1. For each op we have a set of N attention wts where N is the length of input sentence
2. These wts should sum upto 1
3. The context for each time step is the weighted sum of the activations of the previous layer where the weights are the sum of the attention wts for that op
    1. For the first op word C_1 = alpha_1,1 x a_1 + alpha_1,2 x a_2 + .. alpha_1,N x a_N
4. The network on the 2nd layer now takes in each such context and op of previous step and produces the ops one by one

    ![https://i.imgur.com/aGKtoS3.png](https://i.imgur.com/aGKtoS3.png)

### How to compute the attention weights

To recap:

![https://i.imgur.com/LhWkuEL.png](https://i.imgur.com/LhWkuEL.png)

The amount of attention paid of the ${t}'$ th word while generating the t_th op

Also remember that for each op we sum across all ${t}'$, so the sum should be 1

![https://i.imgur.com/seQuIcG.png](https://i.imgur.com/seQuIcG.png)

This is nothing but softmax of the e vectors to make sure that they all sum up to 1

What are these e vectors?

Lets go back to the network we had:

![https://i.imgur.com/LcoKz2u.png](https://i.imgur.com/LcoKz2u.png)

s<t-1> was the hidden state from prev time step

a <t'> was the features we received from time step t'

Intuition: How much attention to pay to the activation at t' kind of depends on what is the own hidden state activation from prev time step. Also for each of the words we look at their features     (a <t'>) for t' = 1... N

We use a small NN for this:

![https://i.imgur.com/LVjX6FF.png](https://i.imgur.com/LVjX6FF.png)

NOTE : here probably all the activations for t' = 1... N are fed into the NN for it to pick out a value of e that essentially serves as the correct attention value to pay for that t'

It turns out that for practical purposes this works really well and the NN pays attention to the correct words of the input sentence

### Consolidated diagram

![https://i.imgur.com/Qd1eam7.jpeg](https://i.imgur.com/Qd1eam7.jpeg)

![https://i.imgur.com/i0Mc9va.jpeg](https://i.imgur.com/i0Mc9va.jpeg)

Complexity: If we have T_x words in input and T_y words in op then for each word T_y we have to compute attention for each of T_x words , so we will need to compute T_x x T_y attention parameters (quadratic costs). For machine translation, where the ip and op sentences are fairly short, this is acceptable

Here we have mainly discussed a machine translation prob where the ip were word and op are also words. This has been applied to Image captioning tasks as well, we use a very similar architecture and ips are the pictures (pixels) and pay attention to only parts of picture at a time to generate the captions

![https://i.imgur.com/GSGl0jA.png](https://i.imgur.com/GSGl0jA.png)

This is an example machine translation tasks, and attention wts are computed bw each pair of words in ip and op sentence (white ones represent higher attention values)

## Transformers: Deep Learning @ VU

Notes on the Deep Learning at VU University Amsterdam course: https://dlvu.github.io/
Accompanying blog: http://peterbloem.nl/blog/transformers

## Part 1 : Self Attention

We have already seen a seq2seq layer in which the ips are a seq of tensors (usually vectors) and produce a seq of vectors as an op

![https://i.imgur.com/dSDwzsp.png](https://i.imgur.com/dSDwzsp.png)

The defining property of these layers is how it can handle different lengths of inputs with the same parameters

RNNs can in theory look indefinitely far back into the seq, but the drawback is that it requires sequential processing, so we can compute a time step op before we have computed for the prev time step

*The self attention model will allow us to have parallel computation and the ability to look at any point in the seq before or after the current op*

### The basics of self attention

![https://i.imgur.com/KZ7A99w.png](https://i.imgur.com/KZ7A99w.png)

Here we have a seq of ip vectors and a seq of op vectors and the basic operation to produce a given op vector is a weighted sum over the ip vectors

So for every op we have 6 wts here, we compute the weighted sum and this gives us op

The trick is that these weights (w_i,j) is NOT a model parameter but a derived value that we compute (learn) from the ip data

Lets understand the basics of self-attention graphically firstp

1. We have a seq of 6 ips and 6 corresponding ops. For now we will look at the computation of y3

    ![https://i.imgur.com/3FKdrJB.png](https://i.imgur.com/3FKdrJB.png)

2. We take the corr ip x3 and for every vector in the ip seq we compute a weight by taking the dot product of x3 with that vector

    ![https://i.imgur.com/aIowfSS.png](https://i.imgur.com/aIowfSS.png)

3. Now we have 6 (wts w31 ... w36) w_i,j : how much attention should the output at i pay to ip at position j
4. We take softmax of all the wts so that they sum upto 1

    ![https://i.imgur.com/5JUAeyi.png](https://i.imgur.com/5JUAeyi.png)

5. We multiply each ip vector by the weight computed and we sum them all up and this gives us the vector y3

    ![https://i.imgur.com/xz9aTWO.png](https://i.imgur.com/xz9aTWO.png)

4. The entire process can be described in these equations:

![https://i.imgur.com/SBaDJwz.png](https://i.imgur.com/SBaDJwz.png)

This process can be easily vectorized as shown below:

![https://i.imgur.com/R4E3lIf.png](https://i.imgur.com/R4E3lIf.png)

/// TODO: example to understand how this vectorization works

Couple of things to note here: 

1. The weight from an input vector at one position to the same position will be the highest (w_i,i from x_i to y_i will be the highest)
    1. Essentially what this means is that this simple self attention is keeping every input vector the same but *mixing in a little bit of the values of the other ip vectors*
    2. We can change this behavior later
2. Simple self attention has no parameters that we can tune, we cannot easily tune the behavior. The behavior is entirely driven by the mechanism that generates these input vectors as the weights are nothing but dot products of these input vectors. For example if we take an embedding layer as ip and stick a self-attention layer on top of it, then the embedding layer will **entirely** drive the behavior of the model
3. Whole self attention is just a matrix multiplication of W by X resulting in Y; W is derived from X

    ![https://i.imgur.com/rf9p3I5.png](https://i.imgur.com/rf9p3I5.png)

    - In the purple path, where we have Y = WX, is purely linear, so gradients will be those of a linear operation, no vanishing gradients
    - In the blue lune there is a SoftMax, so we get non-linearity at the price of potentially vanishing gradients
4. Self attention has no points looking far back into the sequence. In RNNs the further back we go into the seq, more are the computation steps bw ip and op vectors, thats not the case here. That is not the case in self attention, as at heart its nore like a set model, not a sequence model
5. The model currently has no access to the sequential information, we will fix this by encoding the sequential structure into the embeddings
6. Self attention is permutation equivariant, i.e if we first permute → then apply self attention, the result will be same as if we did this in reverse

    p(sa(X)) = sa(p(X))

The self attention mechanism seems to be entirely built using dot products. Let us understand how powerful dot products actually are

### The Power of Dot products

Imagine we have a set of users and movies and a mapping of likes, i.e. which user likes which movies, this is an incomplete set so we have to predict what other movies a user might like

We collect feature vectors for the users and movies and we can simply take the dot product to get a score. The higher the score, more likely is the user to like that movie

![https://i.imgur.com/2q04J3W.png](https://i.imgur.com/2q04J3W.png)

- Dot products intuitively take into account signs of the features, for e.g: if likes_romance is +ve and has_romance is also +ve then tis will inc the score, if likes_romance is -ve this will dec the score, if likes_romance is -ve and has_romance is also -ve then these cancel and the score again increases
- The magnitudes also are handled appropriate, if user is neutral towards romance, likes_romance → 0 then whether or not the movie is highly romantic or not, that term would not contribute much towards the score
- If we cannot build these hand-crafted feature vectors, we can always use embedding vectors, as we know that embedding vectors encode useful features

Lets explore how we use this in Attention

Lets setup a simple sentiment analysis problem where we want to predict a sentiment score based on a review

![https://i.imgur.com/vwHgH9l.png](https://i.imgur.com/vwHgH9l.png)

The embedding layer transforms the input words → input vectors

We have a simple self-attention layer which takes in the embedding vectors and creates an output sequence

The output sequence vectors are summed together to give us a single vector for us to perform the classification (we have to predict whether or not the review is +ve or -ve)

If we did not have the  self attention layer in this model, we would have a model where each ip word could contribute to the op independently of the other words (like a bag-of-words mdoel) and the word "terrible" would probably lead to a -ve sentiment, which would be wrong. Here the meaning of the word "terrible" is kind of inverted by the presence of the word "not". This is where self-attention helps. So we would hope that the model learns that the interaction bw the words "not" and "terrible" is important for the correct result

 What we imagine the model should learn:

![https://i.imgur.com/bi7sdkR.png](https://i.imgur.com/bi7sdkR.png)

When we are determining the op seq for the word "terrible i.e. y_terrible, we would hope that the embedding vector for the word "not" is learned in such a way that it has a low dot product with the embedding vector for the word "terrible". So if the 2 occur together, we can lower the probability that terrible contributes negatively to the score as "not" kind of inverts the meaning of the word "terrible". With one simple self attention layer this might not work as expected, but we will add some modifications later

### Features to add to simple self attention

### 1. Scaled self attention

![https://i.imgur.com/5HHgdw7.png](https://i.imgur.com/5HHgdw7.png)

As the dimensionality of the ip vectors grow, so does the avg size of the dot prod. And that growth is by a fator of sqrt(k) where k is the input dimension. We normalize the dot product which keeps the wts in a certain range. As these wts ate then passed to the softmax this helps in avoiding vanishing gradients in the softmax operation

### 2. Keys, Queries and Values

![https://i.imgur.com/Db5Ezxo.png](https://i.imgur.com/Db5Ezxo.png)

![https://i.imgur.com/xz9aTWO.png](https://i.imgur.com/xz9aTWO.png)

Suppose we are talking computing op y3

y3 = w31 . x1 + w32 . x2 + w33 . x3 + w34 . x4

Where w31 = x3 . x1

w32 = x3. x2 ... and so on...

y3 = (x3 . x1). **x1** + (x3. x2) . **x2** +(x3. x3) . **x3** + (x3. x4) . **x4**

`underline: query, regular: key, bold: value`

Now we basically consider x3 as the query, which is matched against the ips (keys) : x1 →x4

We can imagine like in the prev example that x3 is an encoding which encodes specific features, so if x3 is the encoding for the word "kingdom" it may contain feature info like ["male", "royalty", "people", "ruling", ...]. Imagine a query like 

(blog_blob): How important is the features set: ["male", "royalty", "people", "ruling", ...] interaction with every other input word important for predicting y3

Now when we use this query to compute the dot prod against say x1, which acts as a key here, if the interaction is important, a high wight will be computed which will be multiplied with the corresponding value

Query is like the input vector corr to the current op (y3), which is matched against every other ip vector

x1→x4 are the keys; the vector that the query is matched again 

The values are the the vector in the weighted sum that ultimately provides the op, again x1 →x4

NOTE: here for simple self attention for every op y3 we have a single query x3. This query matches against every word; isn't a single query too simplistic

Some notes

- every key matches the query to a certain extent as determined by the dot prod
- A mixture of all values is returned
- This is basically how the nw can learn associations bw diff parts of the text while generating each op
- **In self attention, keys, queries and values are all from the same set, i.e all derived from the ip embeddings and that is why we call it "self"; also a query at a time for y3 is x3 so again "self"**

![https://i.imgur.com/LBKQqUm.jpeg](https://i.imgur.com/LBKQqUm.jpeg)

### More on Queries, Keys and Values

Again consider the equation for determining y3:

y3 = (x3 . x1). **x1** + (x3. x2) . **x2** +(x3. x3) . **x3** + (x3. x4) . **x4**

`underline: query, regular: key, bold: value`

Every input vector 𝐱i is used in three different ways in the self attention operation:

1. It is compared to every other vector to establish the weights for its own output 𝐲i; so for obtaining the op corr to x3 i.e y3, x3 acts like a query and it is compared with every other input vector
2. It is compared to every other vector to establish the weights for the output of the j-th vector 𝐲j

    So imagine we are generating y3. Now a vector like x2 contributes in the term (x3. x2) . **x2** and if its significant, it will contribute to the overall op, so it is getting compared with every other vector, like a key and `key x query = wt`

3. It is used as part of the weighted sum to compute each output vector once the weights have been established.

These roles are often called the query, the key and the value (we'll explain where these names come from later). In the basic self-attention we've seen so far, each input vector must play all three roles. We make its life a little easier by deriving new vectors for each role, by applying a linear transformation to the original input vector.

### Modifications to self attention

### 1. Key, Query and Value Transformations

![https://i.imgur.com/NZMk9Wo.png](https://i.imgur.com/NZMk9Wo.png)

![https://i.imgur.com/T13Fdka.jpeg](https://i.imgur.com/T13Fdka.jpeg)

![https://i.imgur.com/EDxMcVG.jpeg](https://i.imgur.com/EDxMcVG.jpeg)

### 2. Multi-head attention

![https://i.imgur.com/locrErO.png](https://i.imgur.com/locrErO.png)

The necessity of MHA derives from the idea that diff words relate to each other by diff relations. 

Here the word terrible relates to the words not and too in that they kind of neglect and change the ,meaning of the word terrible

But the relation bw the word terrible and restaurant is completely diff

To allow the nw to model one kind of relation in one self attention layer is feasible but for different relations we will need different self attention layers, same inputs and architecture, but added in parallel to the first layer

Basically we split the self attention into diff heads

1. Start with an input seq
2. Pass this seq through some linear operations to decrease its dimensionality
3. Here we consider a 2-head self attention. W1 and W2 are the transformations

    ![https://i.imgur.com/Je8y8HE.png](https://i.imgur.com/Je8y8HE.png)

4. Each of these reduced dimensional inputs are fed to 2 separate self attentions in parallel
5. Each of the self attention layers have their own K,Q and V transforms. 
6. We get 2 sets of seq vectors out.
7. We concatenate them and pass through a final op transformation (WO) to give us the op seq

    ![https://i.imgur.com/HGTMOLh.png](https://i.imgur.com/HGTMOLh.png)

---

# Attention paper explanation

Credits: https://www.youtube.com/watch?v=qYcy6h1Rkgg

At a very simple, high level we can think of attention as a form of **weighted average pooling**

![https://i.imgur.com/SV0ek7X.png](https://i.imgur.com/SV0ek7X.png)

w1, w2, w3 are like the attention paid to each input item

**Self-attention: when the wts itself are the functions of the inputs** 

The way of calculating attention varies across different papers

![https://i.imgur.com/57ibx9f.jpeg](https://i.imgur.com/57ibx9f.jpeg)

![https://i.imgur.com/PWaFTSw.jpeg](https://i.imgur.com/PWaFTSw.jpeg)

### Attention - the math

Credit: https://www.youtube.com/watch?v=ouLAC55VVOY&t=2319s

**Attention as a hash table + multi headed attention**

![https://i.imgur.com/7cilSKX.jpeg](https://i.imgur.com/7cilSKX.jpeg)

**Representing it as equations**

![https://i.imgur.com/KAhnQVq.jpeg](https://i.imgur.com/KAhnQVq.jpeg)

![https://i.imgur.com/cov12GH.jpeg](https://i.imgur.com/cov12GH.jpeg)

**Adding positional embeddings into the equation:**

![https://i.imgur.com/StLCRdL.jpeg](https://i.imgur.com/StLCRdL.jpeg)

![https://i.imgur.com/zBAxMgg.jpeg](https://i.imgur.com/zBAxMgg.jpeg)

**Transformer (TODO)**

![https://i.imgur.com/Y3yDqCv.jpeg](https://i.imgur.com/Y3yDqCv.jpeg)

## Attention - Michigan Online

Credits: https://www.youtube.com/watch?v=YAgjfMR9R_M

Imagine a seq→seq task we are solving using RNNs 

Here x values are the words in English and y values are the corresponding Spanish translations

Note that the ip and op seq might be of different lengths (here T and T')

In a seq→seq architecture, we have one RNN called the **Encoder,** which would receive the the x vectors  one at a time and produce this sequence of hidden states h_1 through h_t

At every time step we would use this for current neural network formulation f_w that would receive the previous hidden state and the current input vector x and then produce the next hidden state

So we use 1 RNN to process the whole seq of ip vectors. In the diagram we see an "unrolled" diagram meaning that the RNN is copied for each time step but its actually just 1 RNN which processes inputs sequentially

![https://i.imgur.com/V2s8tMY.png](https://i.imgur.com/V2s8tMY.png)

Once we finish processing the ip vectors we want to summarize the contents using 2 vectors here, Also we will be using another separate RNN as the decoder

s_0 : initial hidden state of decoder RNN

C: context vector; this will be fed at each time step to the decoder

Some common practices is to set the context vector to the final hidden state (here h_4)

and also to set the initial hidden state s_0 to 0

The decoder at the first time step receives and input <START> token, prev hidden state and the context vector

So the decoder op hidden state for the first time step s_1 is

`s_1 = decoder(y_0, s_0, C)` 

NOTE  in the diag below it says that the decoder receives h_t-1 instead of s_t-1, which probably means that the decoder receives the prev hidden state from its own perspective, which is s_t-1

In this way successively the op words are generated until a <STOP> token comes

![https://i.imgur.com/fZs93K4.png](https://i.imgur.com/fZs93K4.png)

### Some limitations

The context vector serves the purpose of transferring info bw the encoder and decoder, so its supposed to somehow summarize all the info that the decoder **needs**  and what the encoder **sees**

For long input sequences a vector C of limited length cannot contain that much info

TODO: vanishing grad (blog_blob)

**Idea: use new context vector at each step of the decoder**

We achieve this by using something called Attention

### Getting into Attention

blog_blob

The intuition is that suppose we are predicting the the 3rd hidden state from the decoder, so what are the things that might be important for it?

1. Of course the previous op hidden state (s_t-1)
2. Also each of the ip hidden states might be important to a certain degree
- well , what is that degree - we can simply use some form of weighted average - [proceed to explain a simplified wt avg method as by namvo] - kind of like an intuitive explanation

Below we have the formalized working 

Now what's this certain degree ? i.e. how much attention to give to each ip hidden state given the previous hidden state of the decoder - we simply let a NN figure that out!

So we get a series of wts e31, e32, e33, e34 where e_3i = Fatt(s_2, h_i) for i = 1→4

So we get a set of wts to apply to each hidden state and we can train this nw so that it learns **how much** attention to pay to what input and **when** (when is achievable as we are providing the prev hidden state as well to the NN) - so u can imagine that the NN can **potentially** learn the pattern, when I have seen "xyz" and there is "abc" in the ip I must attend to that

In more technical terms if we look at the above eqn it means that each attn wt is how much does the model think that the hidden state h_i is necessary for predicting the word that comes after the hidden state s_2 (in terms of predicting s_3, the 3rd word)

 Now that we have constructed a set of attention wts for each time step of the decoder, we can summarize that info and feed it in as a context vector at each time step of the decoder

- we apply softmax on the attention wts to convert to a prob distribution
- We take a wt sum of each hidden state (vector) with these attention wts to get a vector with dimenionality same as the hidden state of encoder
- This gives the context vector for that predicting the particular time step of the decoder (in our above example its C3) Note: in the diag its C1 as we are considering predicting the first op word
- Basically as we are predicting each word depending on the word predicted before, we can attend to diff parts of the ip differently

![https://i.imgur.com/1ZX2e3t.png](https://i.imgur.com/1ZX2e3t.png)

 This diagram shows for the first op word, for the second (to generate s_2) similarly we use s_1 and compute attention wts to get C2

(blog_blob) Should we not also use s_0 while calculating C2 - well, s1 = decoder(y0, s0, C1) so the prev hidden states info is already kind of encoded in the immediately prev hidden state

![https://i.imgur.com/NiyTOXY.png](https://i.imgur.com/NiyTOXY.png)

Now since we have diff context vectors for each time step of the decoder, we have solved the problem of bottlenecking for long ip sequences

(blog_blob): prob of enc-dec → attn intuition → attn implementation → attn interpretation → [filler] proceed to building block of transformer → attn as a wt avg → self attn → K,Q,V intuition → multi head attn

### Interpreting the attention wts

![https://i.imgur.com/g45GW56.png](https://i.imgur.com/g45GW56.png)

This is basically a heatmap depicting the wts computed for each input word while predicting each op word

If we look at the 1st 4 words we see that only the diagonal elements are high which means that the one-to-one relationship is meaningful

But then we come to the op words "zone economique europeene" and we can see the off-diagonal elements of the respective english words highlighted

The model has figured out that the order is different for these words and this trend continues 

So using these attention wts we can **explain** how the model is making these decisions

And this is a very direct interpretation, we do not have to do any adversarial attacks/compute permutation importances. Basically no extra computation, we just get these attention wts out that we have to compute anyways and we get the added benefit of interpretatbility

 

### Attention applied to other tasks

Note that the input to attention mechanism is basically some input features and an initial hidden state. Then using the framework we compute a new op hidden state and so on...

So we can very easily apply to any other task in which we have a grid of ip features and we want to produce a sequence of ops. We can apply it to other types of ip data which do not necessarily have to be sequences

We apply it to an image captioning task:

Imagine that we have a grid of feature vectors as the final op of the conv layer. These vectors represent the features at that particular spatial portion of the image, so h11 in the diag represents the features at the top left portion of the image We use the exact same attention mechanism to compute the attn wts, which represent how much attn to pay to each **spatial portion of the image** for generating each op. Each attn wt is a scalar which we store in another grid. Using these attn wts we compute a context vector as before (C1)

![https://i.imgur.com/YfX5JDq.png](https://i.imgur.com/YfX5JDq.png)

Similarly we use S1 to compute new attn wts and C2 to predict "sitting"

![https://i.imgur.com/a3fi56H.png](https://i.imgur.com/a3fi56H.png)

 and we continue ....

![https://i.imgur.com/gN2l5hp.png](https://i.imgur.com/gN2l5hp.png)

This is v similar to the seq→seq task as before

Now again since the attn wts represents which parts of the image to pay attention to as we predict each op word, we can visualize them very easily

![https://i.imgur.com/KsVmmte.png](https://i.imgur.com/KsVmmte.png)

Here we have generated the op caption "A bird flying over a body of water"

At every time step of generating these words we have visualized the attn wts

Observe how the attn initially focuses on the bird and then focuses on the body of water!!

Also note how the transition kind of happens near the word "over" when the model starts to see the lower pixels more, almost as if it were asking "over what??"

In the lower row, we ask the model not to select a weighted recombination of all features, we ask the model to select exactly one feature in the input grid of features

Some other examples:

![https://i.imgur.com/QYexVSe.png](https://i.imgur.com/QYexVSe.png)

**Anytime we want to convert one type of data into another type of data and we want to do it over time, we can use attention mechanism to cause the model to focus on diff parts of the ip while generating each part of the op - very general mechanism**

### Developing the extensions of attention model

Here we basically let a feed forward NN take in the prev hidden state and all the input features and let it assign all the attention wts

This is way better than our prev seq → seq architecture but its still sequential, meaning that we have to compute the current hidden state before computing the next one

![https://i.imgur.com/gk4tnJq.jpeg](https://i.imgur.com/gk4tnJq.jpeg)

![https://i.imgur.com/6NVhwIc.jpeg](https://i.imgur.com/6NVhwIc.jpeg)

![https://i.imgur.com/mRyNcoy.jpeg](https://i.imgur.com/mRyNcoy.jpeg)

Here we have used the prev hidden state as query vectors, given I have seen "" what wts to get in order for me to get the next token

We have seen for a single query, let us now generalize for a set of input query vectors. Note that we are now removing the seq aspect and assuming that we get all the query vectors as ip to the system simultanously

![https://i.imgur.com/PytFZvi.jpeg](https://i.imgur.com/PytFZvi.jpeg)

![https://i.imgur.com/RTCFQcW.jpeg](https://i.imgur.com/RTCFQcW.jpeg)

![https://i.imgur.com/rKuz0OT.jpeg](https://i.imgur.com/rKuz0OT.jpeg)

**Transformations to the input features**

![https://i.imgur.com/4QA62hs.jpeg](https://i.imgur.com/4QA62hs.jpeg)

![https://i.imgur.com/1YJGy1Q.jpeg](https://i.imgur.com/1YJGy1Q.jpeg)

Calculating as before:

![https://i.imgur.com/Dc7O2M2.jpeg](https://i.imgur.com/Dc7O2M2.jpeg)

![https://i.imgur.com/4zKMRHC.jpeg](https://i.imgur.com/4zKMRHC.jpeg)

![https://i.imgur.com/6IQePvg.jpeg](https://i.imgur.com/6IQePvg.jpeg)

[Reading the paper - some notes](https://www.notion.so/Reading-the-paper-some-notes-5d325d7101be4ff2b935596003e920a3)