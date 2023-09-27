# glovpy
Package for interfacing Stanford's C GloVe implementation from Python.

## Installation

Install glovpy from PyPI:

```bash
pip install glovpy
```

Additionally the first time you import glopy it will build GloVe from scratch on your system.

## Requirements
We highly recommend that you use a Unix-based system, preferably a variant of Debian.
The package needs `git`, `make` and a C compiler (`clang` or `gcc`) installed.

Otherwise the implementation is as barebones as it gets, only the standard library and gensim are being used (gensim only for producing KeyedVectors).

## Example Usage
Here's a quick example of how to train GloVe on 20newsgroups using Gensim's tokenizer.

```python
from gensim.utils import tokenize
from sklearn.datasets import fetch_20newsgroups

from glovpy import GloVe

texts = fetch_20newsgroups().data
corpus = [list(tokenize(text, lowercase=True, deacc=True)) for text in texts]

model = GloVe(vector_size=25)
model.train(corpus)

for word, similarity in model.wv.most_similar("god"):
    print(f"{word}, sim: {similarity}")
```

|   word     |   similarity   |
|------------|---------------|
| existence  |  0.9156746864 |
| jesus      |  0.8746870756 |
| lord       |  0.8555182219 |
| christ     |  0.8517201543 |
| bless      |  0.8298447728 |
| faith      |  0.8237065077 |
| saying     |  0.8204566240 |
| therefore  |  0.8177698255 |
| desires    |  0.8094088435 |
| telling    |  0.8083973527 |

## API Reference 

### `class glovpy.GloVe(vector_size, window_size, symmetric, distance_weighting, alpha, min_count, iter, initial_learning_rate, threads, memory)`

Wrapper around the original C implementation of GloVe.

### Parameters

| Parameter                   | Type              | Description                                                                                      | Default          |
|------------------------|-------------------|--------------------------------------------------------------------------------------------------|------------------|
| vector_size            | _int_             | Number of dimensions the trained word vectors should have.                                      | *50*           |
| window_size            | _int_             | Number of context words to the left (and to the right, if symmetric is True).                   | *15*           |
| alpha                  | _float_           | Parameter in exponent of weighting function; default 0.75                                       | *0.75*         |
| symmetric              | _bool_            | If true, both future and past words will be used as context, otherwise only past words will be used. | *True*       |
| distance_weighting     | _bool_            | If False, do not weight cooccurrence count by distance between words. If True (default), weight the cooccurrence count by inverse of distance between the target word and the context word. | *True* |
| min_count              | _int_             | Minimum number of times a token has to appear to be kept in the vocabulary.                       | *5*            |
| iter                   | _int_             | Number of training iterations.                                                                    | *25*           |
| initial_learning_rate  | _float_           | Initial learning rate for training.                                                               | *0.05*         |
| threads                | _int_             | Number of threads to use for training.                                                            | *8*            |
| memory                 | _float_           | Soft limit for memory consumption, in GB. (based on simple heuristic, so not extremely accurate)  | *4.0*           |

### Attributes

| Name | Type | Description |
|------|------|-------------|
| wv   | _KeyedVectors_ | Token embeddings in the form of [Gensim keyed vectors](https://radimrehurek.com/gensim/models/keyedvectors.html). |

### Methods

#### `glovpy.GloVe.train(tokens)`
Train the model on a stream of texts.

| Parameter | Type | Description |
|-----------|------|-------------|
| tokens    | _Iterable[list[str]]_ | Stream of documents in the form of lists of tokens. The stream has to be reusable, as the model needs at least two passes over the corpus. |

### `glovpy.utils.reusable(gen_func)`
Function decorator that turns your generator function into an
iterator, thereby making it reusable.
You can use this if you want to reuse a generator function so that multiple passes can be made.

### Parameters

| Parameter | Type     | Description                                  |
|-----------|----------|----------------------------------------------|
| gen_func  | _Callable_ | Generator function that you want to be reusable. |

### Returns

|  Returns  | Type     | Description                                            |
|-----------|----------|--------------------------------------------------------|
| _multigen | _Callable_ | Iterator class wrapping the generator function. |

### Example usage

Here's how to stream a very long file line by line in a reusable manner.

```python
from gensim.utils import tokenize
from glovpy.utils import reusable
from glovpy import GloVe

@reusable
def stream_lines():
    with open("very_long_text_file.txt") as f:
        for line in f:
            yield list(tokenize(line))

model = GloVe()
model.train(stream_lines())
```
