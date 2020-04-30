import numpy as np


def _tuple_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield tuple(next(i) for i in iters)
    except StopIteration:
        pass


def _list_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield [next(i) for i in iters]
    except StopIteration:
        pass


def _dict_generator(nested_vals):
    iters = {k: iter(nested_generator(v)) for k, v in nested_vals.items()}
    try:
        while True:
            yield {k: next(i) for k, i in iters.items()}
    except StopIteration:
        pass


def nested_generator(nested_vals):
    if isinstance(nested_vals, np.ndarray):
        return nested_vals
    elif isinstance(nested_vals, (list, tuple)):
        if all(isinstance(v, str) for v in nested_vals):
            return nested_vals
        elif isinstance(nested_vals, tuple):
            return _tuple_generator(nested_vals)
        else:
            return _list_generator(nested_vals)
    elif isinstance(nested_vals, dict):
        return _dict_generator(nested_vals)
    else:
        raise TypeError(
            'Unrecognized type for nested_generator: %s'
            % str(type(nested_vals)))
