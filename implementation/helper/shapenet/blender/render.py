import os
from pathlib import PurePosixPath

from helper.shapenet.datareader.reader import DataReader
from graphicUtils.image.objConverter import obj_to_image
from helper.io.json import dict_to_json


def render_data_images(output_path, tmp_path, cat_id, item_id,
                       overwrite, image_count, category_zip_file,
                       blender_path, render_script_path, shape_x, shape_y):
    base_path = os.path.join(output_path, "images", cat_id)
    os.makedirs(base_path, exist_ok=True)
    if not overwrite and os.path.exists(os.path.join(base_path,
                                                     item_id, ".semaphore")):
            print("Skipping processing as already processed")
            return False

    for item in category_zip_file.namelist():
        if item.startswith(str(PurePosixPath(cat_id, item_id))):
            category_zip_file.extract(item, tmp_path)

    object_path = os.path.join(tmp_path, cat_id, item_id, "model.obj")
    obj_to_image(blender_path, render_script_path, object_path,
                 base_path, image_count, shape_x, shape_y,
                 cat_id, item_id)


def render_cat(input_path, output_path, tmp_path, render_script_path, cat_id,
               image_count, shape_x, shape_y, overwrite, reverse,
               fixed_meshes, blender_path, l_range, u_range):

    reader = DataReader(input_path)
    cat_data_list = reader.list_archived_data(cat_id)
    category_zip_file = None
    os.makedirs(tmp_path, exist_ok=True)

    """For partial processing"""
    if u_range == -1:
        cat_data_list = cat_data_list[l_range:]
    else:
        cat_data_list = cat_data_list[l_range:u_range]

    if reverse:
        cat_data_list = cat_data_list[-1::1]

    if fixed_meshes:
        raise NotImplementedError("")
    else:
        category_zip_file = reader.get_zip_file(cat_id)

    """Store additional metadata that
    will be used for training process"""
    metadata = {}
    metadata['shapeX'] = shape_x
    metadata['shapeY'] = shape_y
    metadata['imageCount'] = image_count
    dict_to_json(os.path.join(output_path, "images", cat_id, "metadata.json"),
                 metadata)

    print("Processing started")
    for cat_data in cat_data_list:
        render_data_images(output_path, tmp_path, cat_id, cat_data,
                           overwrite, image_count, category_zip_file,
                           blender_path, render_script_path, shape_x, shape_y)
