from graphicUtils.visualizer.mayaviVisualizer import visualize_mesh
from datasetReader.h5py.meshReader import MeshReader
from mayavi import mlab
import numpy as np

reader = MeshReader("/media/saurabh/e56e40fb-030d-4f7f-9e63-42ed5f7f6c71/preprocessing_new/")
dataset = reader.get_dataset("03948459")
with dataset:
    a = visualize_mesh(np.array(dataset['800cf19b2f7ce07e1ee14c0944aebb52']['vertices']),
                       np.array(dataset['800cf19b2f7ce07e1ee14c0944aebb52']['faces']))
    mlab.show()
