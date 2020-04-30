import os
import numpy as np

from dictionaryDataset.hdf5DataDictionary import Hdf5DataDictionary
from dictionaryDataset.hdf5PersistanceManager import Hdf5PersistanceManager
from enum import Enum
from graphicUtils.mesh.meshUtils import \
        sample_mesh_faces, sample_mesh_faces_normals


class CloudMode(Enum):
    POINTCLOUD = "pointCloud"
    CLOUDNORMAL = "cloudNormal"


class CloudPersistanceManager(Hdf5PersistanceManager):

    def __init__(self, cat_id, input_path, output_path, mode, n_samples):
        self._cat_id = cat_id
        self._mode = mode
        self._input_path = os.path.join(input_path, "mesh",
                                        str(self._cat_id) + ".hdf5")
        self._n_samples = n_samples

        if self._mode == CloudMode.POINTCLOUD.value:
            self._path = os.path.join(output_path, "pointCloud",
                                      str(self._cat_id) + "_point_cloud.hdf5")
        elif self._mode == CloudMode.CLOUDNORMAL.value:
            self._path = os.path.join(output_path, "pointCloud",
                                      str(self._cat_id) + "_cloud_normal.hdf5")
        else:
            print("Defaulted")

    @property
    def source_path(self):
        return self._input_path

    def get_source_dataset(self):

        def map_function(mesh):
            vertices = np.array(mesh['vertices'])
            faces = np.array(mesh['faces'])
            if self._mode == CloudMode.POINTCLOUD.value:
                return sample_mesh_faces(vertices, faces, self._n_samples)
            elif self._mode == CloudMode.CLOUDNORMAL.value:
                p, n = sample_mesh_faces_normals(vertices, faces,
                                                 self._n_samples)
                return dict(points=p, normals=n)

        mesh_dataset = Hdf5DataDictionary(self.source_path)

        with mesh_dataset:
            keys = [k for k, v in mesh_dataset.iter_items()
                    if len(v['faces']) > 0]
        mesh_dataset = mesh_dataset.subset(keys)
        return mesh_dataset.map(map_function)
