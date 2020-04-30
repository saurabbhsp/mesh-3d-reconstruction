import numpy as np
import os
from dictionaryDataset.hdf5PersistanceManager import Hdf5PersistanceManager
from deformations.meshDeformation import get_thresholded_template_mesh
from deformations.utility.deform import get_ffd
from graphicUtils.mesh.meshUtils import sample_mesh_faces


def calculate_ffd(vertices, faces, n=3, n_samples=None):
    if n_samples is None:
        points = vertices
    else:
        print("Sampling mesh face")
        points = sample_mesh_faces(vertices, faces, n_samples)
    dims = (n, ) * 3
    return get_ffd(points, dims)


class FFDPersistanceManager(Hdf5PersistanceManager):

    def __init__(self, base_path, cat_id, edge_length_threshold=None,
                 n_samples=None, n=3):
        self._cat_id = cat_id
        self._edge_length_threshold = edge_length_threshold
        self._n_samples = n_samples
        self._path = os.path.join(base_path, "template_ffd",
                                  str(cat_id)+
                                  "_"+str(self._n_samples)+
                                  "_"+str(self._edge_length_threshold)+".hdf5")
        self._base_path = base_path
        self._n = n
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

    def get_source_dataset(self):
        base = get_thresholded_template_mesh(self._base_path, self._cat_id,
                                             self._edge_length_threshold)

        def map_fn(base):
            vertices, faces = (
                np.array(base[k]) for k in ('vertices', 'faces'))
            b, p = calculate_ffd(vertices, faces, self._n, self._n_samples)
            return dict(b=b, p=p)
        return base.map(map_fn)


def get_template_ffd(base_path, cat_id, edge_length_threshold=None,
                     n_samples=None):

    manager = FFDPersistanceManager(base_path, cat_id, edge_length_threshold,
                                    n_samples)

    if not os.path.exists(manager.path):
        return manager.get_saved_dataset()
    else:
        return manager.get_destination_dataset()
