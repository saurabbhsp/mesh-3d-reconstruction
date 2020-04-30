import os
import subprocess
import shutil

from pathlib import Path
from helper.io.semaphore import create_semaphore


def obj_to_image(blender_path, render_script_path, input_path,
                 output_path, image_count, shape_x, shape_y,
                 cat_id, item_id, remove_tmp_file=True):
    subprocess.call([blender_path,
                     '--background',
                     '--python', render_script_path, '--',
                     '--views', str(image_count),
                     '--shape', str(shape_x), str(shape_y),
                     '--output_folder', str(output_path),
                     '--remove_doubles',
                     '--edge_split',
                     input_path])

    """Write semaphore to indicate completion of process"""
    create_semaphore(os.path.join(output_path, item_id))
    if remove_tmp_file:
        shutil.rmtree(str(Path(input_path).parent))
