from graphicUtils.mesh.meshDataPersistanceManager import MeshPersistanceManager


def generateMeshData(cat_id, input_path, output_path,
                     overwrite=True):
    meshDataPersistanceManager = MeshPersistanceManager(cat_id, input_path,
                                                        output_path)
    meshDataPersistanceManager.save_all(overwrite)
    with meshDataPersistanceManager.get_destination_dataset() as ds:
        empty_mesh = []
        for item_id, mesh in ds.iter_items():
            if len(mesh['faces']) == 0:
                empty_mesh.append(item_id)
        for empty_mesh_id in empty_mesh:
            ds.remove(empty_mesh_id)
