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