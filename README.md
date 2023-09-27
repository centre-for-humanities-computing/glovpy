# glopy
Package for interfacing Stanford's C GloVe implementation from Python.

## Installation

Install glopy from PyPI:

```bash
pip install glopy
```

Additionally the first time you import glopy it will build GloVe from scratch on your system.

## Requirements
We highly recommend that you use a Unix-based system, preferably a variant of Debian.
The package needs `git`, `make` and a C compiler (`clang` or `gcc`) installed.

## Example Usage
Here's a quick example of how to train GloVe on 20newsgroups using Gensim's tokenizer.

```python
from gensim.utils import tokenize
from sklearn.datasets import fetch_20newsgroups

from glopy import GloVe

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
