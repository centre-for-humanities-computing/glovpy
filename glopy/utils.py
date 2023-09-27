import functools
from itertools import islice
from typing import Callable


def reusable(gen_func: Callable) -> Callable:
    """
    Function decorator that turns your generator function into an
    iterator, thereby making it reusable.

    Parameters
    ----------
    gen_func: Callable
        Generator function, that you want to be reusable

    Returns
    ----------
    _multigen: Callable
        Sneakily created iterator class wrapping the generator function
    """

    @functools.wraps(gen_func, updated=())
    class _multigen:
        def __init__(self, *args, limit=None, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
            self.limit = limit
            # functools.update_wrapper(self, gen_func)

        def __iter__(self):
            if self.limit is not None:
                return islice(
                    gen_func(*self.__args, **self.__kwargs), self.limit
                )
            return gen_func(*self.__args, **self.__kwargs)

    return _multigen
