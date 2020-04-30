"""from dictionaryDataset.core import BaseDataSet


class KeyMappedDictionary(BaseDataSet):

    def __init__(self, source_dataset, key_fn, inverse_key_function):
        self._source_dataset = source_dataset
        self._key_fn = key_fn
        self._inverse_key_function = inverse_key_function

    @property
    def is_open(self):
        return self._source_dataset.is_open

    def keys(self):
        if self._inverse_key_function is None:
            raise IOError("Unknown inverse key function")
        else:
            keys = [self._inverse_key_function(k) for k in self._base.keys()
                    if k is not None]
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

    def __remove(self, key):
        self._source_dataset.__remove(self._key_fn(key))

    def __setitem__(self, key, value):
        self._source_dataset.__set_item__(self._key_fn(key), value)"""
