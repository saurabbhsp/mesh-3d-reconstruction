import os
from dictionaryDataset.hdf5PersistanceManager import Hdf5PersistanceManager
from helper.shapenet.datareader.meshReader import get_zip_file_data_dictionary
from graphicUtils.mesh.meshUtils import read_raw_mesh


class MeshPersistanceManager(Hdf5PersistanceManager):

    def __init__(self, cat_id, input_path, output_path):
        self._cat_id = cat_id
        self._input_path = input_path
        self._path = os.path.join(output_path, "mesh",
                                  str(self._cat_id) + ".hdf5")

    def get_source_path(self):
        return os.path.join(self._input_path,
                            '%s.zip' % self._cat_id)

    def get_source_dataset(self):

        def map_function(f):
            vertices, faces = read_raw_mesh(f)[:2]
            return dict(vertices=vertices, faces=faces)

        return get_zip_file_data_dictionary(self._cat_id,
                                            self.get_source_path()).map(
                                            map_function)
