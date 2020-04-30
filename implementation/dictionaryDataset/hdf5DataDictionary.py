import h5py
import os
import numpy as np

from dictionaryDataset.core import PartialDictionarySet


class Hdf5DataDictionary(PartialDictionarySet):

    def __init__(self, path, mode='a'):
        self._path = path
        self._mode = mode
        self._base = None

    @property
    def path(self):
        return self._path

    @property
    def is_open(self):
        return self._base is not None

    def _open_connection(self):
        if self.is_open:
            raise IOError("Data dictionary is already open")
        else:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._base = h5py.File(self._path, self._mode)

    def _close_connection(self):
        if self.is_open:
            """Will call h5py close method"""
            self._base.close()
            self._base = None

    def _save_item(self, group, key, value):
        if isinstance(value, np.ndarray):
            return group.create_dataset(key, data=value)
        elif key == 'attrs':
            if not hasattr(value, 'items'):
                raise ValueError('attrs value must have `items` attr')
            for k, v in value.items():
                group.attrs[k] = v
        elif hasattr(value, 'items'):
            subgroup = None
            try:
                subgroup = group.create_group(key)
                for k, v in value.items():
                    self._save_item(subgroup, k, v)
                return subgroup
            except Exception:
                if subgroup is not None and key in subgroup:
                    del subgroup[key]
                raise
        else:
            raise IOError("Invalid input provided")

    def __setitem__(self, key, value):
        self._save_item(self._base, key, value)

    def __delitem__(self, key):
        del self._base[key]
