import os
from helper.shapenet.datareader.reader import DataReader
from pathlib import PurePosixPath
from graphicUtils.voxels.objConverter import obj_to_binvox


def generateVoxels(input_path, output_path, binvox_executable_path,
                   cat_id, tmp_path, l_range,
                   u_range, voxel_dimensions=32, exact=True, dc=True,
                   aw=True, reverse=False, overwrite=True, pb=True):

        reader = DataReader(input_path)
        cat_data_list = reader.list_archived_data(cat_id)
        category_zip_file = reader.get_zip_file(cat_id)
        os.makedirs(tmp_path, exist_ok=True)
        """For partial processing"""
        if u_range == -1:
            cat_data_list = cat_data_list[l_range:]
        else:
            cat_data_list = cat_data_list[l_range:u_range]

        if reverse:
            cat_data_list = cat_data_list[-1::1]
        print("Processing started")

        for item_id in cat_data_list:
            base_path = os.path.join(output_path, "voxels", cat_id, item_id)
            os.makedirs(base_path, exist_ok=True)
            if not overwrite and os.path.exists(os.path.join(base_path,
                                                             ".semaphore")):
                print("Skipping processing as already processed")
                continue
            for item in category_zip_file.namelist():
                if item.startswith(str(PurePosixPath(cat_id, item_id))):
                    category_zip_file.extract(item, tmp_path)

            object_path = os.path.join(tmp_path, cat_id, item_id, "model.obj")
            obj_to_binvox(object_path, base_path, binvox_executable_path,
                          voxel_dim=voxel_dimensions, dc=dc, exact=exact,
                          aw=aw, pb=pb)
