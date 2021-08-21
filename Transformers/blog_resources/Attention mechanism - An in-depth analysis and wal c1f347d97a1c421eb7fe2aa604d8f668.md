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

As mentioned before, this is a weighted average over the input features where the weights are learned based on the i/p feature and the output generated so far. Thus we have solved the problem of the context vector being a single static vector which was not enough to summarize the input features efficiently, we now have a mechanism to compute the context vector dynamically at each time step.

We have understood the math behind attention mechanism and hopefully you can now appreciate the intuition behind it. In the animation below I have attempted to visualize the entire process. Assume we start right after the Encoder has finished training and given use the set of input features/hidden state vectors $[h_1, h_2, ... h_8]$

[https://youtu.be/iWZWg6uWbJM](https://youtu.be/iWZWg6uWbJM)

1. $[h_1, h_2, ... h_8]$ are the outputs of the encoding process
2. At each time step we compute a set of attention weights using the function $f_{att}$ which takes as inputs:
    1. The hidden state of the decoder
    2. The input hidden state we are computing the attention weight for
3. Using the set of attention weights we compute a weighted average w.r.t the input features to get the context vector for that time step
4. The Decoder takes as inputs:
    1. The context vector we just computed
    2. The previous hidden state
    3. The previous output embedding
5. The decoder outputs a new hidden state which gets mapped to an output token and the entire process repeats

Note that in the animation I have not shown the normalization process for the attention scores for sake of simplicity. 

Another thing worthwhile to note is that the main output of the attention mechanism is the context vector CT which is fed to the decoder at every time step. The decoder network is exactly the same as before. You can choose any architecture for the decoders as long as it can take in the context vector at each time step. So from the next part we will only focus on the output of the attention mechanism and not particularly on the decoder part as that part can change based on what task we are solving for.

We have understood how attention works and why it is needed. We have used a neural network to learn the function $f_{att}$, which is a perfectly reasonable approach but we will see in the next post how we can simplify and generalize this. Also note that we have abstracted the process so that $[h_1, h_2, ..., h_8]$ are in no way constrained to be outputs of an RNN or even textual features - they are simply input features. However we are expected to provide a sequence of vectors $s_0 \rightarrow s_1 \rightarrow ... \rightarrow s_7$, one by one and get the attention weights for that time step. So we cant quite parallelize this process yet - we will see how we can re-frame the architecture to solve for this in the next post. In the next post we will work our way through a set of abstractions at the end of which you can simply imagine **attention** as a  **mechanism**  to simply throw in whenever you have some input features and want to generate outputs in a sequence.

# Part 2

We loosely mentioned some ideas in the last part by which we can abstract some parts of the Attention mechanism and make it more generalizable. In this part we will go over a series of abstractions step by step and also understand the math with examples to build attention as a tool which we can use to attend over any set of input features when we are generating a sequence of outputs.

The flow for this part is hugely inspired from the [] and I have heavily borrowed from the explanations, do watch this lecture series!

## Abstraction 1: The input features

The input features do not need to be sequential, like the output from an encoder RNN like we saw in the previous post, we can have a set of input features and choose to attend over these features.

Lets apply this abstraction to a new task of image captioning . The main motive here is to show you how we can apply attention over an arbitrary set of input features

The task of image captioning is to take an image as input and produce a caption for the image

![https://i.imgur.com/WTo5xmU.png](https://i.imgur.com/WTo5xmU.png)

We will still be using an encoder-decoder based architecture for this problem, but this time the encoder will be a CNN network (typically a stack of CNN, FF NN ) which will take in the image pixels and compute a feature set - a feature set output from a CNN network is a grid of features in the form of a feature map. If you are aware of how CNNs work, then this should be familiar to you. Otherwise simply imagine that the output of the CNN is a grid of numerical values that encode spatial features about the image. At a very high level you can imagine that the CNN has detected specific objects in the image as shown in the diagram below and the numerical values in the highlighted locations correspond to those objects.

 

![https://i.imgur.com/EAL39js.png](https://i.imgur.com/EAL39js.png)

So the output of this CNN network is basically a grid of feature values that encode some useful information about the image that helps the network generate the corresponding captions. Here I have assumed that these features correspond to the objects in the image.

Notice how this is very similar to the previous task in which encoder **RNN** detected a **set** of features about an input **sentence**. Here we have an encoder **CNN** detect a **grid** of features about an input **image.**

We had a **set** of attention wts for the set of input features, similarly here we will build a **grid** of attention wts for the grid of input features.

[CNN attention.mp4](Attention%20mechanism%20-%20An%20in-depth%20analysis%20and%20wal%20c1f347d97a1c421eb7fe2aa604d8f668/CNN_attention.mp4)

Here we have shown for a single context vector $c_t$ which will be fed to the decoder. But they key here is that at each time step this context vector is calculated by attending over different *locations* of the image, so you can imagine that when the model needs to output the word "frisbee" we can the weights highlighted in image [] around the frisbee will be high.

///// interpretation

## Abstraction 2: The Neural Network for computing attention - replace with a simple dot product

![https://i.imgur.com/5iQeov8.png](https://i.imgur.com/5iQeov8.png)

We have mentioned previously that the function $f_{att}$ which computes the attention weights, taking as inputs the hidden state vector and the input features is learned by a feed forward neural network.  What does this neural network do? It takes in the previous hidden state and the input features and computes a set of wts that determine how closely the input features interact with the previous hidden state in order to predict the output - so a loose assumption we can make here is that this NN is like a similarity function bw the previous hidden state and the input features - I know that this seems like an oversimplification but for now just understand that essentially what the NN is doing is figuring out how closely aligned (similar) each input feature is to the hidden state to predict the op.

We have already see how a dot product is an excellent measure of similarity between 2 vectors - so lets try and replace this NN by a simple dot product - this will help reduce computation significantly as we do not need to train the NN to predict each attention weight!

The next sections will be a bit math-heavy but we will walk through an example time step to show how exactly these calculations are made. 

Some notation conventions used:

$$\textup{UPPERCASE } \mathbf{ bold} : \textbf{X} \rightarrow matrix \\\textup{lowercase } \mathbf{ bold} : \textbf{s} \rightarrow vector \\\textup{lowercase } regular : a \rightarrow scalar$$

### Step 1 : Initializing the variables

![https://i.imgur.com/4LCXc0a.png](https://i.imgur.com/4LCXc0a.png)

- Assume we are currently on time step $t=2$, so we have the hidden state $s_2$.                            $s_{xy} : \textup{At time step x, the yth dimension of hidden state/query vector}$
- Assume that the hidden state has a dimensionality of 3 ($D_Q$)
- We assume we have the input features in a grid of shape: $N_X \times D_Q$ (Here 4x3). This means that we have 4 input features, each of dimensionality 3.                                                               $h_{xy} : \textup{The yth dimension of xth input feature}$
- We want to find: for this hidden state $s_2$, what are the weights we should assign to the input features in order to predict the output. So the hidden state is like a query, which is to be applied over the keys, which are the input features.

The task here is to understand how the vector $s_2$ interacts with each input feature. In terms of query and keys, we are trying to understand that *given a hidden state vector (query) how does it interact with each of the input features (keys)* - this is the basic motivation for the **QKV** architecture which is used in Transformers (attn is all u need link)

### Step 2: Computing the raw Attention weights

![https://i.imgur.com/J9kRKBk.png](https://i.imgur.com/J9kRKBk.png)

- Previously as you can see in animation [] we had computed each attention weight using the function $f_{att}$. Now as we are using the dot product, we can simply compute $\boldsymbol{E} = \boldsymbol{s_2}\cdot \mathbf{X^{T}} \\$and get the set of attention weights in a single operation
- $\boldsymbol{E}$ is the set of attention weights over each input feature
- Also note that we have used a new notation $\mathbf{a_{x\bullet}}$ . This is simply a cleaner way to write vectors. What it means is that keeping the first dimension of $\mathbf{a}$ fixed we expand along the second dimension (the dimension where the $\bullet$ is present). Just for clarity, $\mathbf{s_{2\bullet}}\cdot \mathbf{h_{1\bullet}} = s_{21}.h_{11} + s_{22}.h_{12} + s_{23}.h_{13}$. I could have avoided this notation, but if you look at explanations in this field where matrix-vector or  matrix-matrix multiplications are used, then this notation is often used.

### Step 3: Normalizing the raw Attention weights

![https://i.imgur.com/I6QVtBZ.png](https://i.imgur.com/I6QVtBZ.png)

This is a self-explanatory step in which we simply apply the SoftMax operation to convert the raw attention weights to a probability distribution. Note that we now have a set of 4 weights that tells us that in a scale of 1-100 what percentage of attention we should provide to each input feature, given this current query vector - hopefully you can see how interpretable and intuitive this is!

//// scaled dot product? 36:00 of video?

### Step 4: Computing the output vector

![https://i.imgur.com/1kAkeAw.png](https://i.imgur.com/1kAkeAw.png)

- The output vector $o$ is defined as $\boldsymbol{o} = \boldsymbol{A}\cdot \boldsymbol{X}$
- The shape of this vector is same as the previous query vector - we can use this vector as the context vector to be fed into the decoder

To understand more concretely what this output vector looks like lets zoom into one of the terms:

 

![https://i.imgur.com/gJU3NFh.png](https://i.imgur.com/gJU3NFh.png)

- Here we have picked the term $\mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet2}}$ and expanded it - as you can see its simply a weighted average of each of the input features (considering the second dimension).  Remember that the input features are stacked in a grid where the number of rows = number of input features and number of columns = number of dimensions in each input
- Similarly $\mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet1}}$ would be a weighted average of each of the input features along the first dimension and $\mathbf{a_{2\bullet}}\cdot \mathbf{h_{\bullet3}}$ will be the same for the 3rd dimension
- Thus we are considering each input features to a certain extent

## Abstraction 3: Generalizing for multiple query vectors

We have abstracted the attention mechanism in a couple of ways so far:

1. It can work on any type of input features (keys)
2. We have replaced the neural network by the dot product between each key and the query vector to calculate the attention weights
3. But still the query vectors are fed into the attention mechanism one at a time, in the above example we saw for the query vector $s_2$. We want to be able to input a matrix of query vectors and get the outputs out in a single time step.

The process remains largely the same as the previous one, lets go over each step as before

### Step 1 : Initializing the variables

Assume we have $N_Q$ query vectors each of dimension $D_Q$ . So we essentially have a Query matrix of shape ($N_Q \times D_Q$). Here $N_Q = 2, D_Q = 3$. The input features matrix $X$ is same as before

![https://i.imgur.com/ddMOeZD.png](https://i.imgur.com/ddMOeZD.png)

### Step 2: Computing the raw Attention weights

![https://i.imgur.com/Njp6D8X.png](https://i.imgur.com/Njp6D8X.png)

1. The equations are same as we saw before, $\boldsymbol{E} = \boldsymbol{Q}\cdot \mathbf{X^{T}}$ allows us to compute the attention weights for each query vector w.r.t each key in one single operation
2. The interpretation of each raw attention weight $e_{xy}$ is same as we saw before. The only difference is that $\boldsymbol{E}$ is a matrix instead of a vector as we are calculating for multiple query vectors

### Step 3: Normalizing the raw Attention weights

$\boldsymbol{E}$ is the attention weight matrix. Each row of $\boldsymbol{E}$ contains $N_X$ raw un-normalized weights, each weight corresponding to an input feature. There are $N_Q$ such rows in $\boldsymbol{E}$

Now, for a particular query, each input feature gets assigned a weight and these weights need to be normalized, so we should apply the SoftMax operation on $\boldsymbol{E}$ along each row

![https://i.imgur.com/WVQ3gW0.png](https://i.imgur.com/WVQ3gW0.png)

### Step 4: Computing the output matrix

![https://i.imgur.com/PfaAv8X.png](https://i.imgur.com/PfaAv8X.png)

In this step we compute the output matrix. We will also call this the value matrix. 

The computation is same as before and we get 2 rows as we had 2 queries. For each row we have 3 dimensions (this is the number of dimensions in each query)

As before by zooming into one of the terms of the output we can see that each output value element is essentially nothing but a weighted average and that it considers a specific dimension of each of the 4 input features (in this case the second dimension)

Notice how the equations all remain the same, by simply stacking the query vectors as a matrix we can ensure that we can compute all values simultaneously.