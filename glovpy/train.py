import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List
from urllib.parse import quote_plus, unquote_plus

from gensim.models import KeyedVectors

from glovpy.utils import reusable


def collect_vocab(corpus: Iterable[bytes], directory: Path, min_count: int):
    """Returns vocab file name."""
    exec_path = (
        Path.home().joinpath(".glopy", "build", "vocab_count").absolute()
    )
    vocab_path = directory.joinpath("vocab.txt")
    with open(vocab_path, "wb") as vocab_file:
        process = subprocess.Popen(
            [exec_path, "-min-count", str(min_count)],
            stdin=subprocess.PIPE,
            stdout=vocab_file,
        )
        for text in corpus:
            process.stdin.write(text)
        process.stdin.close()
        process.wait()


def count_cooccurrences(
    corpus: Iterable[bytes],
    directory: Path,
    symmetric: bool,
    window_size: int,
    memory: float,
    distance_weighting: bool,
):
    """Returns cooccurrence file."""
    exec_path = Path.home().joinpath(".glopy", "build", "cooccur").absolute()
    vocab_path = directory.joinpath("vocab.txt").absolute()
    cooccurrence_path = directory.joinpath("coccurr.bin").absolute()
    params = [
        "-vocab-file",
        vocab_path,
        "-memory",
        str(memory),
        "-window-size",
        str(window_size),
        "-symmetric",
        str(int(symmetric)),
        "-distance-weighting",
        str(int(distance_weighting)),
    ]
    with open(cooccurrence_path, "wb") as cooccurrence_file:
        process = subprocess.Popen(
            [exec_path, *params],
            stdin=subprocess.PIPE,
            stdout=cooccurrence_file,
        )
        for text in corpus:
            process.stdin.write(text)
        process.stdin.close()
        process.wait()


def shuffle_cooccurrences(directory: Path, memory: float):
    exec_path = Path.home().joinpath(".glopy", "build", "shuffle").absolute()
    cooccurrence_path = directory.joinpath("coccurr.bin").absolute()
    shuffle_path = directory.joinpath("shuffle_coccurr.bin")
    with open(shuffle_path, "wb") as shuffle_file:
        with open(cooccurrence_path, "rb") as cooccurrence_file:
            subprocess.run(
                [exec_path, "-memory", str(memory)],
                stdin=cooccurrence_file,
                stdout=shuffle_file,
            )


def train_glove(
    directory: Path,
    vector_size: int,
    threads: int,
    iter: int,
    initial_learning_rate: float,
    alpha: float,
):
    exec_path = Path.home().joinpath(".glopy", "build", "glove").absolute()
    shuffle_path = directory.joinpath("shuffle_coccurr.bin").absolute()
    out_file = directory.joinpath("vectors").absolute()
    vocab_path = directory.joinpath("vocab.txt").absolute()
    parameters = [
        "-vocab-file",
        vocab_path,
        "-vector-size",
        str(vector_size),
        "-threads",
        str(threads),
        "-iter",
        str(iter),
        "-eta",
        str(initial_learning_rate),
        "-alpha",
        str(alpha),
        "-input-file",
        shuffle_path,
        "-save-file",
        out_file,
    ]
    subprocess.run([exec_path, *parameters])


@reusable
def encode_tokens(tokens: Iterable[List[str]]) -> Iterable[bytes]:
    """Encodes all documents into bytes.
    All tokens in a document get url-quoted, then joined by spaces,
    a newline is added and then it is encoded to ascii.

    Reusable iterables remain reusable with this generator.
    """
    for document in tokens:
        encoded_tokens = [quote_plus(token) for token in document]
        encoded_line = (" ".join(encoded_tokens) + "\n").encode("ascii")
        yield encoded_line


def get_keyed_vectors(directory: Path) -> KeyedVectors:
    """Converts vectors to Gensim KeyedVectors
    and stores them in 'plutarch_vectors/'"""
    vectors_path = directory.joinpath("vectors.txt")
    return KeyedVectors.load_word2vec_format(
        vectors_path, binary=False, no_header=True
    )


def unquote_keyed_vectors(kv: KeyedVectors) -> KeyedVectors:
    """Unquotes keys in the keyed vectors produced by the model."""
    res = KeyedVectors(vector_size=kv.vector_size, count=len(kv))
    for key in kv.index_to_key:
        res.add_vector(unquote_plus(key), kv.get_vector(key))
    return res


class GloVe:
    """Wrapper around the original C implementation of GloVe.

    Parameters
    ----------
    vector_size: int, default 50
        Number of dimensions the trained word vectors should have.
    window_size: int, default 15
        Number of context words to the left
        (and to the right, if symmetric is True).
    alpha: float, default 0.75
        Parameter in exponent of weighting function; default 0.75
    symmetric: bool, default True
        If true, both future and past words will be used as context,
        otherwise only past words will be used.
    distance_weighting: bool, default True
        If False, do not weight cooccurrence count by distance between words
        If True (default), weight the cooccurrence count by inverse of
        distance between the target word and the context word.
    min_count: int, default 5
        Minimum number of times a token has to appear to be kept in
        the vocabulary.
    iter: int, default 25
        Number of training iterations.
    initial_learning_rate: float, default 0.05
        Initial learning rate for training.
    threads: int, default 8
        Number of threads to use for training.
    memory: float, default 4.0
        Soft limit for memory consumption, in GB.
        (based on simple heuristic, so not extremely accurate)

    Attributes
    ----------
    wv: KeyedVectors
        Token embeddings in form of Gensim keyed vectors.
    """

    def __init__(
        self,
        vector_size: int = 50,
        window_size: int = 15,
        symmetric: bool = True,
        distance_weighting: bool = True,
        alpha: float = 0.75,
        min_count: int = 5,
        iter: int = 25,
        initial_learning_rate: float = 0.05,
        threads: int = 8,
        memory: float = 4.0,
    ):
        self.vector_size = vector_size
        self.window_size = window_size
        self.symmetric = symmetric
        self.distance_weighting = distance_weighting
        self.alpha = alpha
        self.min_count = min_count
        self.iter = iter
        self.initial_learning_rate = initial_learning_rate
        self.threads = threads
        self.memory = memory

    def train(self, tokens: Iterable[List[str]]):
        """Train the model on a stream of texts.

        Parameters
        ----------
        tokens: iterable of list of str
            Stream of documents in the form of list of tokens.
            The stream has to be reusable, as the model needs at least two
            passes over the corpus.
        """
        encoded = encode_tokens(tokens)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            print("Collecting Vocabulary...")
            collect_vocab(encoded, directory, min_count=self.min_count)
            print("Collecting cooccurrences...")
            count_cooccurrences(
                encoded,
                directory,
                symmetric=self.symmetric,
                window_size=self.window_size,
                memory=self.memory,
                distance_weighting=self.distance_weighting,
            )
            print("Shuffling cooccurrences...")
            shuffle_cooccurrences(directory, memory=self.memory)
            print("Training model...")
            train_glove(
                directory,
                vector_size=self.vector_size,
                threads=self.threads,
                iter=self.iter,
                initial_learning_rate=self.initial_learning_rate,
                alpha=self.alpha,
            )
            self.wv = get_keyed_vectors(directory)
            self.wv = unquote_keyed_vectors(self.wv)
