import logging

from glopy.install import install_glove, is_installed
from glopy.train import GloVe

if not is_installed():
    logging.info(
        "Glove is not installed on your system yet, installing from source."
    )
    install_glove()

__all__ = ["GloVe"]
