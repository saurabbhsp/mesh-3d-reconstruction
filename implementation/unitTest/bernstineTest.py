from helper.shapenet.shapenetMapper import desc_to_id
from deformations.FFD import get_template_ffd
from deformations.meshDeformation import get_thresholded_template_mesh
from mayavi import mlab
import numpy as np
from graphicUtils.visualizer.mayaviVisualizer import visualize_mesh, visualize_point_cloud



ds = get_template_ffd("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c711/preprocessing_new", desc_to_id("pistol"),
                      edge_length_threshold=None,  n_samples=16384)

key = "1f646ff59cabdddcd810dcd63f342aca"
with ds:
    b = np.array(ds[key]['b'])
    p = np.array(ds[key]['p'])

mesh_dataset = get_thresholded_template_mesh("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c711/preprocessing_new", desc_to_id("pistol"),
                      None)

with mesh_dataset:
    f = np.array(mesh_dataset[key]['faces'])
    v_orignal = np.array(mesh_dataset[key]['vertices'])

# print(b)
# visualize_mesh(v_orignal, f)
# mlab.show()

visualize_mesh(np.matmul(b, p), f)
mlab.show()

visualize_point_cloud(np.matmul(b, p))
mlab.show()

# from deformations.FFD import calculate_ffd

# b, p = calculate_ffd(v_orignal, f)
# isualize_mesh(np.matmul(b, p), f)
# mlab.show()
