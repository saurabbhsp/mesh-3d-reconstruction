from dictionaryDataset.hdf5DataDictionary import Hdf5DataDictionary
import os


class MeshReader(object):

    def __init__(self, base_path):
        self._base_path = base_path

    def get_dataset(self, cat_id):
        return Hdf5DataDictionary(os.path.join(self._base_path, "mesh",
                                  "%s.hdf5" % (cat_id)))
