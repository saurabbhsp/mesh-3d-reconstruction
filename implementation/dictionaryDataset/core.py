# from dictionaryDataset.keyMappedDictionary import KeyMappedDictionary
# from dictionaryDataset.mappedDictionary import MappedDictionary

"""Abstract class. Will be
extended for other datasets"""


class BaseDataSet(object):

    connections = None

    def __init__(self):
        self._connections = set()

    @property
    def connections(self):
        if not hasattr(self, '_connections') or self._connections is None:
            self._connections = set()
        return self._connections

    @property
    def is_open(self):
        """To check if the dictionary is accessable"""
        return True

    """Standard dictionary methods"""
    def keys(self):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    def __getitem__(self, key):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    def __setitem__(self, key, value):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    def __delitem__(self, key):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    """Standard implemented methods"""
    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def iter_items(self):
        return [(k, self[k]) for k in self.keys()]

    def __len__(self):
        return len(self.keys())

    def get(self, key):
        return self[key]

    """Following methods are written so
    that the dictionary can be used in with block"""

    def _open_connection(self):
        pass

    def open_connection(self, client):
        connections = self.connections
        if len(connections) == 0:
            self._open_connection()
        connections.add(client)

    def _close_connection(self):
        pass

    def close_connection(self, client):
        connections = self.connections
        if connections is not None:
            connections.remove(client)
        if len(connections) == 0:
            self._close_connection()

    def open(self):
        self.open_connection(self)

    def close(self):
        self.close_connection(self)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def to_dict(self):
        return {k: v for k, v in self.iter_items()}

    def save_dataset(self, source_dataset, overwrite=True):
        if not self.is_open:
            raise IOError("Dataset is not open")
        keys = source_dataset.keys()
        if not overwrite:
            keys = [key for key in keys if key not in self]

        for key in keys:
            if key in self:
                self.__delitem__(key)
            self.__setitem__(key, source_dataset[key])

    def map_keys(self, key_fn, inverse_key_function=None):
        return KeyMappedDictionary(self, key_fn, inverse_key_function)

    def map(self, map_fn):
        return MappedDictionary(self, map_fn)

    def subset(self, keys):
        return SubDictionary(self, keys)


class PartialDictionarySet(BaseDataSet):
    """
    Partially implemented abstract class
    """
    def __init__(self, base_dataset):
        if base_dataset is None:
            raise ValueError("Base dataset cannot be of None")
        self._base = base_dataset
        super(PartialDictionarySet, self).__init__()

    @property
    def is_open(self):
        if hasattr(self._base, 'is_open'):
            return self._base.is_open
        else:
            return True

    def _open_connection(self):
        self._base.open_connection(self)

    def _close_connection(self):
        self._base.close_connection(self)

    """Implemented methods"""
    def keys(self):
        return self._base.keys()

    def __getitem__(self, key):
        return self._base[key]

    def __setitem__(self, key, value):
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]


class SubDictionary(PartialDictionarySet):

    def __init__(self, base_dataset, keys):
        self._keys = frozenset(keys)
        super(SubDictionary, self).__init__(base_dataset)

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        return self._base[key]

    def __setitem__(self, key, value):
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]

    def subset(self, keys):
        return SubDictionary(self._base, keys)


class MappedDictionary(PartialDictionarySet):

    def __init__(self, base_dataset, map_fn):
        self._map_fn = map_fn
        super(MappedDictionary, self).__init__(base_dataset)

    def __contains__(self, key):
        return key in self._base

    def __getitem__(self, key):
        return self._map_fn(self._base[key])

    def __len__(self):
        return len(self._base)

    def subset(self, keys):
        return self._base.subset(keys).map(self._map_fn)


class KeyMappedDictionary(BaseDataSet):

    def __init__(self, source_dataset, key_fn, inverse_key_function):
        self._source_dataset = source_dataset
        self._key_fn = key_fn
        self._inverse_key_function = inverse_key_function
        super(KeyMappedDictionary, self).__init__()

    @property
    def is_open(self):
        return self._source_dataset.is_open

    def keys(self):
        if self._inverse_key_function is None:
            raise IOError("Unknown inverse key function")
        else:
            keys = [self._inverse_key_function(k) for k in
                    self._source_dataset.keys()]
            keys = [k for k in keys if k is not None]
            return keys

    def __len__(self):
        return self._source_dataset.__len__()

    def __getitem__(self, key):
        mapped_key = self._key_fn(key)
        try:
            return self._source_dataset[mapped_key]
        except KeyError:
            raise KeyError("No such key found")

    def __contains__(self, key):
        return self._source_dataset.__contains__(self._key_fn(key))

    def _open_connection(self):
        self._source_dataset._open_connection()

    def _close_resource(self):
        self._source_dataset._close_connection()

    def __delitem__(self, key):
        self._source_dataset.__delitem__(self._key_fn(key))

    def __setitem__(self, key, value):
        self._source_dataset.__setitem__(self._key_fn(key), value)
