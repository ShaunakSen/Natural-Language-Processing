# Understanding Transformers

Paper link: https://arxiv.org/pdf/1706.03762.pdf

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