from helper.shapenet.shapenetMapper import desc_to_id
from deformations.FFD import get_template_ffd
from deformations.meshDeformation import get_thresholded_template_mesh
from mayavi import mlab
import numpy as np
from graphicUtils.visualizer.mayaviVisualizer import visualize_mesh, visualize_point_cloud
from deformations.FFD import calculate_ffd

def permute_xyz(x, y, z, order='xyz'):
    _dim = {'x': 0, 'y': 1, 'z': 2}
    data = (x, y, z)
    return tuple(data[_dim[k]] for k in order)


ds = get_template_ffd("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c711/preprocessing_new", desc_to_id("pistol"),
                      edge_length_threshold=None)

key = "1f646ff59cabdddcd810dcd63f342aca"
with ds:
    b = np.array(ds[key]['b'])
    p = np.array(ds[key]['p'])

mesh_dataset = get_thresholded_template_mesh("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c711/preprocessing_new", desc_to_id("pistol"),
                      None)

with mesh_dataset:
    f = np.array(mesh_dataset[key]['faces'])
    v_orignal = np.array(mesh_dataset[key]['vertices'])



data = permute_xyz(*p.T, order='xyz')
mlab.points3d(*data, scale_factor=0.01)
x, y, z = permute_xyz(*v_orignal.T, order='xyz')
mlab.triangular_mesh(
    x, y, z, f, color=(0, 0, 1), representation='wireframe')
mlab.show()

rand = np.random.rand(64,3)
data = permute_xyz(*(p+rand).T, order='xyz')
mlab.points3d(*data, scale_factor=0.01)
x, y, z = permute_xyz(*np.matmul(b, p+rand).T, order='xyz')
mlab.triangular_mesh(
    x, y, z, f, color=(0, 0, 1), representation='wireframe')
mlab.show()
