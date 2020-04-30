from graphicUtils.visualizer.mayaviVisualizer import visualize_point_cloud
from graphicUtils.visualizer.mayaviVisualizer import visualize_normals
from datasetReader.h5py.cloudReader import CloudReader
from mayavi import mlab
import numpy as np

reader = CloudReader("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c71/preprocessing_new/")
dataset = reader.get_dataset("03948459", "pointCloud")
with dataset:
    visualize_point_cloud(
                     np.array(dataset['800cf19b2f7ce07e1ee14c0944aebb52']),
                     scale_factor=0.005, color=(0, 1, 0))
    mlab.show()

#dataset = reader.get_dataset("03948459", "cloudNormal")
#with dataset:
#    a = visualize_normals(
#            np.array(dataset['139ecc9b5ec25ea77a9772e56db4cd44']['points']),
#            np.array(dataset['139ecc9b5ec25ea77a9772e56db4cd44']['normals']),
#            scale_factor=0.005, color=(1, 0, 1))
#    mlab.show()
