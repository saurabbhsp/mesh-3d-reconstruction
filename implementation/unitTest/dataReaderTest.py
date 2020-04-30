from helper.shapenet.shapenetMapper import desc_to_id
from helper.shapenet.datareader.reader import DataReader


path = "/data/Training Data/ShapeNetCore.v1"
category = "plane"
data_reader = DataReader(path)
print(data_reader.list_archived_data(desc_to_id(category)))
