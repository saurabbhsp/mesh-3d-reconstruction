import subprocess
import os
import shutil

from helper.io.semaphore import create_semaphore
from pathlib import Path


def obj_to_binvox(obj_path, binvox_path, binvox_executable_path, voxel_dim=32,
                  bounding_box=(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
                  pb=True, exact=True, dc=True, aw=True, remove_tmp_file=True):

    if not os.path.isfile(obj_path):
        raise IOError("No obj file found")

    """binvox executable does not take any output path
    and outputs by default in same path where the input
    is present. We will let in execute it and then
    copy it to proper path"""

    temp_output_path = obj_path[:-4] + '.binvox'

    if os.path.isfile(temp_output_path):
        raise IOError("Already exists")

    args = [binvox_executable_path, '-d', str(voxel_dim), '-bb']
    args.extend([str(b) for b in bounding_box])
    for condition, flag in ((pb, '-pb'), (exact, '-e'),
                            (dc, '-dc'), (aw, '-aw')):
        if condition:
            args.append(flag)

    args.append(obj_path)
    subprocess.call(args)
    if not os.path.isfile(temp_output_path):
        raise IOError("Failed to generate the bin voxels")
    os.rename(temp_output_path,
              os.path.join(binvox_path, os.path.basename(temp_output_path)))
    """Write semaphore to indicate completion of process"""
    create_semaphore(binvox_path)
    if remove_tmp_file:
        shutil.rmtree(str(Path(obj_path).parent))
