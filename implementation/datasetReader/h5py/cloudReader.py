import os
from graphicUtils.pointCloud.pointcloudPersistanceManager import CloudMode
from dictionaryDataset.hdf5DataDictionary import Hdf5DataDictionary


class CloudReader(object):

    def __init__(self, base_path):
        self._base_path = base_path

    def get_dataset(self, cat_id, mode):
        if mode == CloudMode.POINTCLOUD.value:
            return Hdf5DataDictionary(os.path.join(
                                      self._base_path,
                                      "pointCloud",
                                      "%s_point_cloud.hdf5" % (cat_id)))
        elif mode == CloudMode.CLOUDNORMAL.value:
            return Hdf5DataDictionary(os.path.join(
                                        self._base_path,
                                        "pointCloud",
                                        "%s_cloud_normal.hdf5" % (cat_id)))
        else:
            print("Invalid mode selected")
