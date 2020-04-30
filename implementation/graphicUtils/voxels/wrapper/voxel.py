import numpy as np
from graphicUtils.voxels.voxelUtils import read_header
from helper.io.runningLengthEncoding import rle_to_dense, rle_to_sparse, \
                                            sorted_gather_1d, dense_to_rle, \
                                            sparse_to_rle

"""Abstract class voxels"""


class Voxels(object):

    def __init__(self, dims, translate=(0, 0, 0), scale=1):

        """Convert to 3 X 3 representation"""
        if isinstance(dims, int):
            self._dims = (dims,) * 3
        elif len(dims) != 3:
            raise ValueError('dims must have 3 elements.')
        else:
            self._dims = tuple(dims)

        self.translate = np.array(translate)
        self.scale = scale

    @staticmethod
    def read_file(fp):
        """The same file pointer is used
        it will read forward all the headers"""
        dims, translate, scale = read_header(fp)
        rle_data = np.frombuffer(fp.read(), dtype=np.uint8)
        """Need to convert the running length encoding"""
        return RleVoxels(rle_data, dims, translate, scale)

    def save(self, path):
        with open(path, 'w') as fp:
            self.save_to_file(fp)

    def save_to_file(self, fp):
        dims = self.dims
        translate = self.translate
        scale = self.scale
        fp.write('#binvox 1\n')
        fp.write('dim ' + ' '.join(map(str, dims)) + '\n')
        fp.write('translate ' + ' '.join(map(str, translate)) + '\n')
        fp.write('scale ' + str(scale) + '\n')
        fp.write('data\n')
        fp.write((chr(d) for d in self.rle_data()))

    @property
    def dims(self):
        return self._dims

    def to_dense(self):
        return DenseVoxels(
            self.dense_data(), self.translate, self.scale)

    def to_sparse(self):
        return SparseVoxels(
            self.sparse_data(), self.dims, self.translate, self.scale)

    def to_rle(self):
        return RleVoxels(self.rle_data(), self.dims)

    def rle_data(self):
        raise NotImplementedError('Abstract method')

    def dense_data(self, fix_coords=False):
        raise NotImplementedError('Abstract method')

    def sparse_data(self, fix_coords=False):
        raise NotImplementedError('Abstract method')


"""Running length encoding voxels"""


class RleVoxels(Voxels):
    def __init__(self, rle_data, dims, translate=(0, 0, 0), scale=1):
        self._rle_data = rle_data
        super(RleVoxels, self).__init__(dims, translate, scale)

    def rle_data(self):
        return self._rle_data

    def dense_data(self, fix_coords=False):
        rle_data = self._rle_data
        data = rle_to_dense(rle_data)
        assert(data.dtype == np.bool)
        data = data.reshape(self.dims)
        if fix_coords:
            data = np.transpose(data, (0, 2, 1))
        return data

    def sparse_data(self, fix_coords=False):
        indices = rle_to_sparse(self._rle_data)
        dims = self.dims
        d2 = dims[2]
        d1 = dims[1]*d2
        i = indices // d1
        kj = indices % d1
        k = kj // d2
        j = kj % d2
        if fix_coords:
            return i, k, j
        else:
            return i, j, k

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            x, y, z = indices
            indices = x, z, y
        indices = np.ravel_multi_index(indices, self.dims)
        order = np.argsort(indices)
        ordered_indices = indices[order]
        ans = np.empty(len(order), dtype=np.bool)
        ans[order] = tuple(self._sorted_gather(ordered_indices))
        return ans

    def _sorted_gather(self, ordered_indices):
        return sorted_gather_1d(self._rle_data, ordered_indices)


class DenseVoxels(Voxels):
    def __init__(self, dense_data, translate=(0, 0, 0), scale=1):
        self._dense_data = dense_data
        super(DenseVoxels, self).__init__(dense_data.shape,
                                          translate, scale)

    def rle_data(self):
        return np.array(tuple(
            dense_to_rle(self._dense_data.flatten())), dtype=np.uint8)

    def dense_data(self, fix_coords=False):
        return self._dense_data

    def sparse_data(self, fix_coords=False):
        i, k, j = np.where(self._dense_data)
        if fix_coords:
            return i, j, k
        else:
            return i, k, j

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            i, j, k = indices
        else:
            i, k, j = indices
        return self._dense_data[i, k, j]


class SparseVoxels(Voxels):
    def __init__(self, sparse_data, dims, translate=(0, 0, 0), scale=1):
        self._sparse_data = sparse_data
        super(SparseVoxels, self).__init__(dims, translate, scale)

    def rle_data(self):
        i, k, j = self._sparse_data
        indices = np.ravel_multi_index((i, k, j), self.dims)
        return sparse_to_rle(indices, np.prod(self.dims))

    def dense_data(self, fix_coords=False):
        dims = self.dims
        if fix_coords:
            dims = dims[0], dims[2], dims[1]
            i, k, j = self._sparse_data
        else:
            i, j, k = self._sparse_data
        data = np.zeros(dims, dtype=np.bool)
        data[i, j, k] = True
        return data

    def sparse_data(self, fix_coords=False):
        i, k, j = self._sparse_data
        if fix_coords:
            return i, j, k
        else:
            return i, k, j

    def gather(self, indices, fix_coords=False):
        if fix_coords:
            i, j, k = indices
        else:
            i, k, j = indices
        dims = self.dims
        indices_1d = np.ravel_multi_index((i, k, j), dims)
        sparse_1d = set(np.ravel_multi_index(self._sparse_data, dims))
        return np.array([i1d in sparse_1d for i1d in indices_1d], np.bool)
