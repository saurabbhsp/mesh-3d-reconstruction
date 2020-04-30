from datasetReader.h5py.meshReader import MeshReader
from templateManager.shapenet.shapenetTemplateManager import get_template_ids


def get_template_mesh(base_path, cat_id):
    meshReader = MeshReader(base_path)
    meshData = meshReader.get_dataset(cat_id)
    return meshData.subset(get_template_ids(cat_id))
