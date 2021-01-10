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