# Attention in Deep Learning - Intuition and in-depth analysis

## A quick overview of the Encoder-Decoder architecture

In this section we will be taking a high level view on the popular encoder-decoder architecture and get a basic idea on how it works. What has this got to do with Attention - the limitations of this architecture inspired the attention mech and we will understand why these limitations occur. 

The encoder decoder architecture has 5 main parts that we want to talk about

### 1. The input sequence

For purposes of this task let us consider we are translating English to French. Lets assume that the sentence we want to translate is "Economic growth has slowed down in recent years." Then the input sequence is basically the words of the sentence. Now NNs can't process raw text so we need to encode them in some numeric format. So each word should become a vector of numbers that we can input to the network. *Why a vector, and not a scalar - each word has multiple characteristics and vector allows for that info to be captured*. Of course we can use one hot encoding for this purpose but a far better approach is to use word embeddings that can capture the similarities between the words. *In the example below we see 3 embeddings for the words "king", "man" and "queen" - explain similarities in features*

So the input sequence is basically a set of vector representations for each word 

//// diagram for word embeddings

### 2. Encoder stack of NNs

A common misconception is that encoder decoder architectures always use rnns for encoding.but we can use an nn which can take in a set of inputs and compute a set of features from it. And since all NNs are feature extrators we can use any nn for this. We will see later in the post a use of CNN as the encoder to extract features from images. Here we will stick to RNNs as it will allow us to process sequential input. 

For simplicity consider that the encoder comprises a single RNN block which processes each input token one by one

```bash
new_hidden_state = RNN(input_token_at_that_time_step, prev_hidden_state)
h_T = RNN(i_T, h_{T-1})
```

//// enter animation here - diag1.2

### 3. Hidden state

Thus the final op of this encoder block is h_T of **any specified dimension** that basically summarizes all important information from the input sequence. This is often referred to as the "context" vector as, for a well trained network this vector summarizes all useful information (features) from the input sentence that can be used by the decoder. Notice how we loosely refer to this context vector as a set of features of the input text, so essentially the encoder network acts as a feature extractor for the input.

Explain figure: [https://arxiv.org/pdf/1409.3215.pdf](https://arxiv.org/pdf/1409.3215.pdf)

### 4. Decoder stack of NNs

Now that we have a hidden state that has learned the important aspects (features) of the input sentence, we need a mechanism that can process these sequence of features and produce the outputs.

Again its not necessary to use RNNs but since we are dealing with sequential data, for this task it makes sense to use a stack of RNNs as the decoder module. For simplicity we consider that we have only one RNN as the decoder

The RNN decoder unit takes as inputs: 

- previous output word embedding
- previous hidden state
- the hidden state h_T - the encoder output

Using these inputs

/// simple block diagram - diag1.3 

```bash
new_hidden_sate = RNN_decoder(prev_op_word_vector, prev_hidden_state, h_T)
```

Let us understand what information each of these inputs provides to the decoder

- previous output word embedding - given that the prev word generated was xyz what should be the next word?
- previous hidden state - as each new_hidden_sate, is computed using prev_hidden_state as we saw in diag1.3, this input captures info about all the outputs which have been generated so far. So given I have seen [token1, token2, tokken3] as op what should be the next token
- h_T - this gives info about the entire input sequence - this vector has summarized the input text and serves as a kind of "context" vector

So, to summarize at each step the decoder mainly tries to compute "given that I have generated the op [token1, token2, tokken3,..] and given that the prev word I generated was [...] and given all the input features, what should be the next hidden state?"

### 5. Output sequence

How to go from a hidden state to an op word?

Again, as we had the english words converted to word vectors we have the French word vectors as well. Given a new hidden state from the op decoder we compute the dot product with each of the op vectors. 

/// diag1.4

The dot product bw 2 vectors is a scalar that can be interpreted as high when the vectors are closely aligned (the features are similar) and small when the vectors are dissimilar

So we get a set of N scores if there are N possible op French words

Instead of picking out the word corresponding to the max score we first run a softmax on the scores to bring them to the range 0→1 so that we can interpret them as probabilities. The inituitive reason for doing this is so that we can interpret the results for e.g: "the translation of x is x2 with 85% probability"

//// diag1.5

## Limitations of this architecture

We have developed this architecture in light of the task of converting English Sentences to French and it seems to be a great solution, and it really is! In fact, ENc -dec architectures are extensively used by Google Translate and we all know how great it is!

However if we consider tasks like text summarization or question-answering systems like chatbots which have to process and remember information from a large piece of text in order to process outputs, the limitations of this architecture become apparent. The main problem is with h_T which is a vector of fixed dimensionality that is supposed to somehow encode all relevant info from the input sequence into this latent space. For translating short pieces of text this is acceptable but for encoding really large input texts, this method fails. Think of Google assistant or Siri from a few years back. It was great at understanding simple queries and answering them but it could not carry out a long conversation as we did not have way of preserving long-term dependencies yet. Of course we can scale up the dimension of this context vector to preserve more information but that will increase the training time and since we are using a sequential nw like RNNs we cant even parallelize it as at each time step we need as input the op of prev time step. So simply increasing the dimensionality is not a feasible soln

//// diag for drop in performance [https://arxiv.org/pdf/1409.1259.pdf](https://arxiv.org/pdf/1409.1259.pdf)

note_to_self: looks really good till this - need some great visualizations though - DONE!

 

## Building towards Attention

Ok, so far we have understood how ED arch works and established that our main limitation is that a single context vector is not enough to efficiently encode all input information of long sequences. 

*Well, the logical solution is to simply have multiple context vectors and feed them into the decoder at each time step.*

Also, Remember that in the ED architecture we had only placed emphasis on the last hidden state of the encoder h_T and discarded all previous hidden states.

The encoder actually computes hidden states h1 , h2 ... hT which are basically the feature extractors of the ip seq 

//// diag1.2

Instead of discarding all this info we can use these hidden states for the input vectors - remember again that we can loosely assume that these are features extracted from the input vectors. Also the purpose of the context vector is to somehow give us information on the input context

Thus the context vector that will be passed to the decoder at time step t should be influenced by:

1. The previous hidden state of the decoder 
2. The hidden states of the encoder (h1 ... hT)

context vector at time step t = f(h1, h2, ... hT, s_t-1)

Again, if we think logically each of the input hidden states are not really equally important for generating each op. Depending on the op we are generating, we need to assign a certain weight (attention) to each of the input features. The wt that we assign to each input feature depending on the op we have generated so far is called **attention**

```bash
context vector at time step t = f(h1, h2, ... hT, s_t-1)  ---1 
context vector at time step t = f(w_1t.h1, w_2t.h2, ... wTt.hT, s_t-1) ---2
where w_it = f_att(hi, s_t-1) ---3

```

Note how equation 2 simply looks like a wt average, where we assign different wts on the diff input features based on the outputs generated (s_t-1 encodes info of the prev outputs generated). It simply tries to figure out how important each of the input features are necessary for predicting the word that comes after the hidden state s_t-1.

New thing is f_att in eqn 3. The task of this function is to come up with a scalar wt depending on the input feature and the previous decoder hidden state. How to derive this function? We simply let a feed-forward neural net figure this out and train it  alongside the network

//// Explain the process *one example time step from diag 2.1 - 2.5

Thus at every time step of the decoder, we have a separate context vector, which is capable of placing attention on individual input vectors while generating the op

So even if we have extremely long ip seq we no longer need to create one vector to summarize everything, we can simply choose to attend over the important parts depending on the current word being generated by the decoder. Thus we have solved the main drawback of encoder decoder architectures

//// explain the entire process in one animation diag step by step

So to sum up, we received a set of hidden states (input features) from the encoder NN and at each step, while generating the output, we placed attention on the input hidden states as well as the output at the previous time steps. So the requirement of the attention mechanism is actually:

- a set of input features we want to attend over
- a set of outputs that we want to generate over time by attending over these different inputs

We have seen how to have applied it to a seq→ seq task but we can very easily apply it to other tasks

### Abstractions

As we concluded the requirements for applying attention to a task is when we have a set of input features that we want to attend over and we want to generate a sequence of outputs.

Thus we are at a great stage to formalize the first abstraction

Abstraction 1 : The input features do not need to be sequential, like the op from an RNN, we can have a set of input features and choose to attend over these features

Lets apply this abstraction to a new task of image captioning  

The following example is based on the excellent video by : []

The main motive here is to show you how we can apply attention over an arbitrary set of input features - it is not necessary that these are the outputs of an RNN or any sequential NN

**The task**

The task is basically to get an image and compute a caption, for example

/// picture of an example - [https://i.imgur.com/WTo5xmU.png](https://i.imgur.com/WTo5xmU.png)

**Overall process**

The way we can design a simplistic system for this is to have a CNN network (typically a stack of CNN, FF NN ) take in the image pixels and compute a feature set - a feature set output from a CNN network is a grid of features in the form of a feature map. If you are aware of how CNNs work, then this should be familiar to you. Otherwise simply imagine that the output of the CNN is a grid of numerical values that encode spatial features about the image. 

/// diag 3.1

Notice how this is very similar to the previous task in which encoder **RNN** detected a **set** of features about an input **sentence**. Here we have an encoder **CNN** detect a **grid** of features about an input **image**

We had a set of attention wts for the set of input features, similarly here we will build a grid of attention wts for the grid of input features

//// see video from 20:35 and explain this process

//// animation/diag explaining the process

Abstraction 2 : The NN for computing attention - replace with a simple dot product

Here we basically let a feed forward NN take in the prev hidden state and all the input features and let it assign all the attention wts. What does this NN do? It takes in the previous hidden state and the input features and computes a set of wts that determine how closely the input features interact with the previous hidden state in order to predict the output - so a loose assumption we can make here is that this NN is like a similarity function bw the previous hidden state and the input features - I know that this seems like an oversimplification but for now just understand that essentially what the NN is doing is figuring out how closely aligned (similar) each input feature is to the hidden state to predict the op (how? - by applying the transformations and the V transformation as well that will determine what to return, not just the similarity score)

The next big thing for us to understand is that we dont really need a complex NN to figure out the similarity between 2 vectors - a dot product is excellent at this!

//// diag 3.2 - explain dot prod as similarity measure

To understand the process, let us zoom into one time step from the fig [], so imagine that we have already generated the first 2 words (hence we have the hidden state s2 available) and we are generating the 3rd word

- Assume that the hidden state has a dimensionality of 3 (Dx)
- Assume that we have 4 input features, each of dimensionality 3, so the matrix of input features is 4x3 (Nx x Dx)

The task here is to understand how the vector s2 interacts with each input feature. In terms of query and keys, we are trying to understand that given a hidden state vector (query) how does it interact with each of the input features (keys) - this is the basic motivation for the QKV architecture

 

### Interpretation

- here interpret both the translation and image captioning task

### Abstraction

start by motivation of how we can replace NN → dot prod while calculating attn wt

Question while calculating attn wt: for the query vector, how relevant is each ip vector (key) - this is answered by a dot product very easily

Notice how gradually encoder hidden states → ip features ...

TO DO : ADD ANIMATION INDEXES 

Here we have calculated the attention weights by a simple nn but there are a lot of different ways of calculating it. The transformer architecture and also perceiver uses a form of qkv calculation which is highly transferable and parallelizable - we will understand this in the next section 

In traditional arch, the context is same for every time step and only the prv hidden state is diff. Say the NN got the first op wrong.. then the entire error kind of propagates and since the context is same at each step, the NN has no way to correct

### Appendix: All equations

$$\boldsymbol{X} = \begin{bmatrix}h_{11} & h_{12}  & h_{13}\\ h_{21} & h_{22}  & h_{23}\\h_{31} & h_{32}  & h_{33} \\h_{41} & h_{42}  & h_{43}\end{bmatrix}_{4\times 3} \boldsymbol{s_2} = \begin{bmatrix}
s_{21} & s_{22} & s_{23}
\end{bmatrix}_{1\times 3}$$

$$\boldsymbol{E} = \begin{bmatrix}
s_{21} & s_{22} & s_{23}
\end{bmatrix}_{1\times 3} \cdot \begin{bmatrix}
h_{11} & h_{21}  & h_{31} & h_{41} \\ 
h_{12} & h_{22}  & h_{32} & h_{42} \\ 
h_{13} & h_{23}  & h_{33} & h_{43} 
\end{bmatrix}_{3\times 4} \\ \\

\therefore \boldsymbol{E} = \begin{bmatrix}
\mathbf{s_{2\bullet}}\cdot \mathbf{h_{1\bullet}} & \mathbf{s_{2\bullet}}\cdot \mathbf{h_{2\bullet}} & \mathbf{s_{2\bullet}}\cdot \mathbf{h_{3\bullet}} & \mathbf{s_{2\bullet}}\cdot \mathbf{h_{4\bullet}}
\end{bmatrix}_{1\times 4}\\

\textup{where }\mathbf{h_{x\bullet}} = \begin{bmatrix}
h_{x1} & h_{x2} & h_{x3} & ...
\end{bmatrix} \\ \\

\textup{we can simplify } \mathbf{E} \textup{ as:} \\

\boldsymbol{E} = \begin{bmatrix}
e_{21} & e_{22} & e_{23} & e_{24} 
\end{bmatrix}_{1\times 4}\\

$$

$$\begin{bmatrix}e_{21}\\ e_{22}\\ e_{23}\\ e_{24}\end{bmatrix} \overset{softmax}{\rightarrow}\begin{bmatrix}a_{21}\\ a_{22}\\ a_{23}\\ a_{24}\end{bmatrix}$$

$$\boldsymbol{A} =  \begin{bmatrix}
a_{21} & a_{22} & a_{23} & a_{24} 
\end{bmatrix}_{1\times 4}\\ \\

\boldsymbol{s_3} = \boldsymbol{A}_{1\times 4} \cdot \boldsymbol{X}_{4\times 3} \\
= \begin{bmatrix}
a_{21} & a_{22} & a_{23} & a_{24} 
\end{bmatrix}_{1\times 4} \cdot \begin{bmatrix}h_{11} & h_{12}  & h_{13}\\ h_{21} & h_{22}  & h_{23}\\h_{31} & h_{32}  & h_{33} \\h_{41} & h_{42}  & h_{43}\end{bmatrix}_{4\times 3} \\
= \begin{bmatrix}
\mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet1}} & \mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet2}} & \mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet3}}
\end{bmatrix}_{1\times 3}\\

\textup{where }\mathbf{h_{\bullet x}} = \begin{bmatrix}
h_{1x} & h_{2x} & h_{3x} & ...
\end{bmatrix} \\ \\$$

$$\texttt{zooming into one single term:}\\\\
\mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet2}} = \mathbf{a_{2\bullet}}\cdot \begin{bmatrix}
h_{12}\\ 
h_{22}\\ 
h_{32}\\ 
h_{42}
\end{bmatrix}\\\\

\sim \texttt{considers second feature of all inputs}$$