import numpy as np
import os
from dictionaryDataset.hdf5PersistanceManager import Hdf5PersistanceManager
from templateManager.templateMesh import get_template_mesh
from graphicUtils.mesh.edgeSplitter import EdgeSplitter
from templateManager.shapenet.shapenetTemplateManager import get_template_ids


class SplitMeshPersistanceManager(Hdf5PersistanceManager):

    def __init__(self, path, cat_id, edge_length_threshold=None,
                 initial_threshold=None):
        self._path = path
        self._cat_id = cat_id
        self._edge_length_threshold = edge_length_threshold
        self._initial_threshold = initial_threshold

        if initial_threshold is not None and \
           initial_threshold <= edge_length_threshold:
            raise ValueError("Initial threshold should be " +
                             "greater than edge threshold")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    @property
    def path(self):
        return os.path.join(self._path, "splitMesh",
                            str(self._edge_length_threshold),
                            '%s.hdf5' % self._cat_id)

    def get_source_dataset(self):
        template_mesh = get_thresholded_template_mesh(
                                                    self._path,
                                                    self._cat_id,
                                                    None
                                                    )

        def map_function(mesh):
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            edgeSplitter = EdgeSplitter(vertices, faces)
            edgeSplitter.split_to_threshold(self._edge_length_threshold)
            return dict(vertices=np.array(edgeSplitter._vertices),
                        faces=np.asarray(list(edgeSplitter._faces)))


        return template_mesh.map(map_function)


def get_thresholded_template_mesh(path, cat_id, threshold):
    if threshold is None:
        return get_template_mesh(path, cat_id)
    else:

        manager = SplitMeshPersistanceManager(path, cat_id,
                                              threshold)
        if not os.path.exists(manager.path):
            dataset = manager.get_saved_dataset()
        else:
            print(manager.path)
            dataset = manager.get_destination_dataset()
        return dataset.subset(get_template_ids(cat_id))
