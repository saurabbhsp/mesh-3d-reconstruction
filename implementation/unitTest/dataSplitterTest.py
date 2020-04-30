from datasetSplitter.shapenet.shapenetTrainTestSplitter import Splitter
from helper.shapenet.shapenetMapper import desc_to_id

source_path = "/data/Training_Data/ShapeNetCore.v1"
destination_path = "/data/output"
cat_id = desc_to_id("car")

splitter = Splitter(destination_path, source_path, cat_id,
                    0.8, 0.1, 0.1, replace=False)

print(splitter.train_set)
