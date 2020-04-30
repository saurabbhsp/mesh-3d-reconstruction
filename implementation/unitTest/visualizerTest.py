from graphicUtils.visualizer.mayaviVisualizer import visualize_mesh
from graphicUtils.visualizer.mayaviVisualizer import visualize_point_cloud
from graphicUtils.visualizer.mayaviVisualizer import visualize_normals
from graphicUtils.visualizer.mayaviVisualizer import visualize_voxels
import h5py
import numpy as np
from mayavi import mlab
from graphicUtils.voxels.wrapper.voxel import DenseVoxels
# data = h5py.File("/data/preprocessing/mesh/03948459.hdf5")
# data = h5py.File("/data/preprocessing/pointCloud/03948459_point_cloud.hdf5")
# data = h5py.File("/data/preprocessing/pointCloud/03948459_cloud_normal.hdf5")

# v, f = (np.array(data['1374e4af8a3ed46ea6766282ea8c438f'][k]) for k in ('vertices', 'faces'))
# p = np.array(data['1374e4af8a3ed46ea6766282ea8c438f'])
# n, p = (np.array(data['1374e4af8a3ed46ea6766282ea8c438f'][k]) for k in ('normals', 'points'))

# a = visualize_mesh(v, f)
# a = visualize_point_cloud(p, color=(0, 1, 0), scale_factor=0.005)
# a = visualize_normals(p, n, color=(1, 1, 0), scale_factor=0.005)
# mlab.show()



"""For voxels"""
filePath = "/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c711/preprocessing_new/voxels/03948459/800cf19b2f7ce07e1ee14c0944aebb52/model.binvox"
voxel = DenseVoxels.read_file(open(filePath, 'rb')).to_dense()

a = visualize_voxels(voxel.dense_data(), color=(0, 0, 1), scale_factor = 1)
mlab.show()
