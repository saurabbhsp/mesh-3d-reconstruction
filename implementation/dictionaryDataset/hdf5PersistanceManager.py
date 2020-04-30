from dictionaryDataset.hdf5DataDictionary import Hdf5DataDictionary
from dictionaryDataset.persistanceManager import PersistanceManager


class Hdf5PersistanceManager(PersistanceManager):

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        return self._path

    def get_destination_dataset(self, mode='a'):
        return Hdf5DataDictionary(self.path, mode)
