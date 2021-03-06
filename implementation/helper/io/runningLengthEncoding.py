"""
Helper methods
for running length encoding
"""
import numpy as np


def rle_to_dense(rle_data):
    values, counts = rle_data[::2], rle_data[1::2]
    return np.repeat(values.astype(np.bool), counts)


def rle_to_sparse(rle_data):
    indices = []
    it = iter(rle_data)
    index = 0
    try:
        while True:
            value = next(it)
            counts = next(it)
            end = index + counts
            if value == 1:
                indices.append(np.arange(index, end, dtype=np.int32))
            index = end
    except StopIteration:
        pass
    indices = np.concatenate(indices)
    return indices


def dense_to_rle(dense_data):
    data_iter = iter(dense_data)
    try:
        count = 0
        value = next(data_iter)
        while True:
            count += 1
            next_val = next(data_iter)
            if next_val != value or count == 255:
                yield value
                yield count
                count = 0
                value = next_val
    except StopIteration:
        if count > 0:
            yield value
            yield count


def _repeated(count):
    while count > 255:
        yield 255
        count -= 255
    if count > 0:
        yield count


def sparse_to_rle(indices, length):
    index_iter = iter(indices)
    try:
        last = next(index_iter)
        if last != 0:
            yield 0
            yield last
        count = 1
        while True:
            n = next(index_iter)
            if n == last + count:
                count += 1
            else:
                for c in _repeated(count):
                    yield 1
                    yield c
                # 0 block
                for c in _repeated(n - last - count):
                    yield 0
                    yield c
                last = n
                count = 1

    except StopIteration:
        for c in _repeated(count):
            yield 1
            yield c
        for c in _repeated(length - n - 1):
            yield 0
            yield c


def sorted_gather_1d(raw_data, ordered_indices):
    data_iter = iter(raw_data)
    index_iter = iter(ordered_indices)
    index = next(index_iter)
    start = 0
    while True:
        while start <= index:
            try:
                value = next(data_iter)
                start += next(data_iter)
            except StopIteration:
                raise IndexError(
                    'Index %d out of range of raw_values length %d'
                    % (index, start))
        try:
            while index < start:
                yield value
                index = next(index_iter)
        except StopIteration:
            break


def _get_contiguous_regions_1d(
        data_iter, max_index, i=0, start_val=0):
    start_index = 0
    vals = []
    try:
        while i < max_index:
            val = next(data_iter)
            n = next(data_iter)
            # print(val, start_val, start_index, i)
            if val != start_val:
                if val == 1:
                    start_index = i
                else:
                    if start_index != i:
                        vals.append((start_index, i))
                    start_index = i
                start_val = val
            i += n
        if start_val == 1:
            vals.append((start_index, max_index))
        done = False
    except StopIteration:
        done = True
    return vals, i, start_val, done


def get_contiguous_regions_1d(rle_data, max_index=np.inf):
    return _get_contiguous_regions_1d(iter(rle_data), max_index)


def get_contiguous_regions_2d(rle_data, dim):
    """
    Get the start/end position of contiguous occupied regions from 2D rle_data.

    Args:
        rel_data: iterable of run-length-encoded values
        dim: last dimension
    Returns:
        Iterable of list of shape (ni, 2), where ni is the number of contiguous
            regions for the ith 1st dimension in rle_data.
    """
    start_val = 0
    start_index = 0
    data_iter = iter(rle_data)
    done = False
    while not done:
        vals, i, start_val, done = _get_contiguous_regions_1d(
            data_iter, dim, start_index, start_val)
        start_index = i - dim
        yield vals


def get_contiguous_regions(rle_data, dims):
    """
    Get the start/end position of contiguous occupied regions from rle_data.

    Args:
        rle_data: iterable of run-length-encoded values
        dims: N-tuple of dimensions
    Returns:
        dims[:-1] + (n, 2) jagged array, where n is a jagged index.
    """
    if len(dims) != 3:
        raise NotImplementedError('Could be generalized, but currently not.')
    u, v, w = dims
    uv = u*v
    ret = np.empty((uv,), dtype=np.object)

    for i, vals in enumerate(get_contiguous_regions_2d(rle_data, w)):
        if len(vals) > 0:
            ret[i] = np.array(vals)
        else:
            ret[i] = np.zeros((0, 2), dtype=np.int32)
    for j in range(i, uv):
        ret[j] = np.zeros((0, 2), dtype=np.int32)
    return ret.reshape((u, v))


def reduce_rle_sum(rle_data):
    s = 0
    it = iter(rle_data)
    try:
        while True:
            value = next(it)
            n = next(it)
            s += n*value
    except StopIteration:
        return s


def sample_occupied_indices(rle_data, n_samples):
    import random
    if n_samples > 0:
        s = reduce_rle_sum(rle_data)
        ns = random.sample(range(s), n_samples)
        ns.sort()
        rle_index = 0
        count = 0
        rle_iter = iter(rle_data)
        for n in ns:
            while n >= count:
                value = next(rle_iter)
                c = next(rle_iter)
                if value:
                    count += c
                rle_index += c
            diff = count - n
            yield rle_index - diff
