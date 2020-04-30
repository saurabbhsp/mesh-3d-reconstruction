from helper.io.json import json_file_reader

_shapenet_id_desc = json_file_reader("/home/saurabh/Documents/project/imageReconstruction/implementation/config/shapenetMapper/" +
                                     "shapenetMapper.json")

_shapenet_desc_id = {v: k for k, v in _shapenet_id_desc.items()}


def desc_to_id(desc):
    return _shapenet_desc_id[desc]
