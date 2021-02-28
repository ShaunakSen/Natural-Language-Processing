# Understanding Transformers

---

Seq2seq and basic attention: Andrew ng+ jay alamar + [https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)

Attention for transformer: Waterloo + jay alamar

Transformer: yannik + jay alamar

[https://www.youtube.com/watch?v=KmAISyVvE1Y](https://www.youtube.com/watch?v=KmAISyVvE1Y)

[https://www.youtube.com/watch?v=6l1fv0dSIg4](https://www.youtube.com/watch?v=6l1fv0dSIg4)

[https://www.youtube.com/watch?v=ah7_mfl7LD0](https://www.youtube.com/watch?v=ah7_mfl7LD0)

---

Paper link: https://arxiv.org/pdf/1706.03762.pdf

Notes on the resources:

- [https://www.youtube.com/watch?v=iDulhoQ2pro](https://www.youtube.com/watch?v=iDulhoQ2pro)
- [https://www.youtube.com/watch?v=SMZQrJ_L1vo](https://www.youtube.com/watch?v=SMZQrJ_L1vo)
- [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

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

Video by AndrewNG: https://www.youtube.com/watch?v=SysgYptB19

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

Complexity: If we have T_x words in input and T_y words in op then for each word T_y we have to compute attention for each of T_x words , so we will need to compute T_x x T_y attention parameters (quadratic costs). For machine translation, where the ip and op sentences are fairly short, this is acceptable

Here we have mainly discussed a machine translation prob where the ip were word and op are also words. This has been applied to Image captioning tasks as well, we use a very similar architecture and ips are the pictures (pixels) and pay attention to only parts of picture at a time to generate the captions

![https://i.imgur.com/GSGl0jA.png](https://i.imgur.com/GSGl0jA.png)

This is an example machine translation tasks, and attention wts are computed bw each pair of words in ip and op sentence (white ones represent higher attention values)

## Transformers: Deep Learning @ VU

Notes on the Deep Learning at VU University Amsterdam course: https://dlvu.github.io/

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

Lets understand the basics of self-attention graphically first

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
    1. Essentially what this means is that this simple self attention is keeping every input vector the same but mixing in a little bit of the values of the other ip vectors
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