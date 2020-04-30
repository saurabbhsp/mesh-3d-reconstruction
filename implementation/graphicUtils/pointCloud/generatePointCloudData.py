from graphicUtils.pointCloud.pointcloudPersistanceManager import \
                            CloudPersistanceManager


def generatePointCloud(cat_id, source_path,
                       destination_path, mode, n_samples):

        manager = CloudPersistanceManager(cat_id, source_path,
                                          destination_path, mode, n_samples)
        manager.save_all()
