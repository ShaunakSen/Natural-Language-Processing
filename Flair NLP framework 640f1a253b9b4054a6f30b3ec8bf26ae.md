# Flair NLP framework

[Flair%20NLP%20framework%20640f1a253b9b4054a6f30b3ec8bf26ae/N19-4010.pdf](Flair%20NLP%20framework%20640f1a253b9b4054a6f30b3ec8bf26ae/N19-4010.pdf)

The original paper, link: [https://www.aclweb.org/anthology/N19-4010.pdf](https://www.aclweb.org/anthology/N19-4010.pdf)

## Key idea

FLAIR attempts to create a simple, unified interface for all word embeddings as well as arbitrary combinations of embeddings

```python
# init sentence 
sentence = Sentence('I love Berlin ')
```

Each Sentence is instantiated as a list of Token objects, each of which represents a word and has fields for tags (such as part-of-speech or named entity tags) and embeddings (embeddings of this word in different embedding spaces).

Embeddings are the core concept of FLAIR. Each embedding class implements either the `TokenEmbedding` or the `DocumentEmbedding` interface for word and document embeddings respectively. Both interfaces define the `.embed()` method to embed a Sentence or a list of Sentence objects into a specific embedding space

The simplest examples are classic word embeddings, such as GLOVE or FASTTEXT. Simply instantiate one of the supported word embeddings and call .embed() to embed a sentence:

```python
# init GloVe 
embeddings glove = WordEmbeddings('glove ')
# embed sentence 
glove.embed(sentence)
```

Here, the framework checks if the requested GLOVE embeddings are already available on local disk. If not, the embeddings are first downloaded. Then, GLOVE embeddings are added to each Token in the Sentence. 

Note that all logic is handled by the embedding class, i.e. it is not necessary to run common preprocessing steps such as constructing a vocabulary of words in the dataset or encoding words as onehot vectors. Rather, each embedding is immediately applicable to any text wrapped in a Sentence object.

As noted in the introduction, FLAIR supports a growing list of embeddings such as hierarchical character features (Lample et al., 2016), ELMo embeddings (Peters et al., 2018a), ELMo transformer embeddings (Peters et al., 2018b), BERT embeddings (Devlin et al., 2018), byte pair embeddings (Heinzerling and Strube, 2018), Flair embeddings (Akbik et al., 2018) and Pooled Flair embeddings. 

See Table 1 for an overview. Importantly, all embeddings implement the same interface and may be called and applied just like in the WordEmbedding example above. For instance, to use BERT embeddings to embed a sentence, simply call:

```python
# init BERT embeddings 
bert = BertEmbeddings ()
# embed sentence 
bert.embed(sentence)
```

![https://i.imgur.com/8x0bNMy.png](https://i.imgur.com/8x0bNMy.png)

Summary of word and document embeddings currently supported by FLAIR. Note that some embedding types are not pre-trained; these embeddings are automatically trained or fine-tuned when training a model for a downstream task.

All models supported by FLAIR: [https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md)

Specific FLAIR embeddings and recommended approach: [https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md](https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)