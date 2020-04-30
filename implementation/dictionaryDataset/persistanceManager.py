from dictionaryDataset.core import BaseDataSet

"""
Bridges two different
dictionary datasets so that data from source dataset
can be copied to the destination dataset.
"""


class BridgeDataSet(BaseDataSet):

    def __init__(self, src, dest):
        if hasattr(src, "iter_items") and hasattr(src, "__getitem__"):
            self._src = src
            self._dest = dest
        else:
            raise Exception("Source should have iter_items and" +
                            " __getitem__ methods")

    @property
    def src(self):
        return self._src

    @property
    def dest(self):
        return self._dest

    def unsaved_keys(self):
        return [key for key in self._src.keys()
                if key not in self._dest.keys()]

    def keys(self):
        return self._src.keys()

    def __contains__(self, key):
        return self._src.__contains__(key)

    def __getitem__(self, key):
        if not self._src.__contains__(key):
            raise ValueError("No such key found")
        if self._dest.__contains__(key):
            return self._dest[key]
        else:
            value = self._src.__getitem__(key)
            self._dest.__setitem__(key, value)
            return value

    def subset(self, keys):
        src = self.src.subset(keys)
        return BridgeDataSet(src, self.dest)

    def save_all(self, overwrite=True):
        self._dest.save_dataset(self._src, overwrite)

    def _open_connection(self):
        self._src._open_connection()
        self._dest._open_connection()

    def _close_connection(self):
        self._src._close_connection()
        self._dest._close_connection()


class PersistanceManager(object):
    """Source dataset"""
    def get_source_dataset(self):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    """Destination dataset"""
    def get_destination_dataset(self, mode='a'):
        raise NotImplementedError("Method should be " +
                                  "extended in derived class")

    def get_bridged_dataset(self, mode='a'):
        return BridgeDataSet(self.get_source_dataset(),
                             self.get_destination_dataset(mode))

    def save_all(self, overwrite=True):
        with self.get_bridged_dataset() as ds:
            ds.save_all(overwrite)

    def get_saved_dataset(self, mode='a'):
        self.save_all()
        return self.get_destination_dataset(mode)
