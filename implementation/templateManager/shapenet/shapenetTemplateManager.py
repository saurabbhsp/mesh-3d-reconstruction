from helper.io.json import json_file_reader

_template_config = json_file_reader("/home/saurabh/Documents/project/imageReconstruction/implementation/config/templates/" +
                                    "shapenet/templates.json")


def get_template_ids(cat_id):
    return _template_config[cat_id]
