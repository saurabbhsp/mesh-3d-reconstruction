import zipfile
from dictionaryDataset.core import BaseDataSet


class ZipFileDataSet(BaseDataSet):

    def __init__(self, path, mode='a'):
        self._path = path
        self._mode = mode
        self._zip_file = None
        self._keys = None

    @property
    def path(self):
        return self._path

    @property
    def mode(self):
        return self._mode

    @property
    def is_open(self):
        return self._zip_file is None

    def _open_connection(self):
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self._path, self._mode)
        else:
            raise IOError("Connection already open")

    def _close_connection(self):
        if self._zip_file is None:
            raise IOError("Connection already closed")
        self._zip_file.close()
        self._zip_file = None

    def keys(self):
        if self._keys is None:
            """Keys are immutable"""
            self._keys = frozenset(self._zip_file.namelist())
        return self._keys

    def __getitem__(self, key):
        return self._zip_file.open(key)
