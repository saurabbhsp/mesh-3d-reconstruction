import os
from dictionaryDataset.zipfileDataDictionary import ZipFileDataSet


def get_zip_file_data_dictionary(cat_id, zip_path):

    def key_fn(item_id):
        return os.path.join(cat_id, item_id, 'model.obj')

    def inverse_key_function(path):
        sub_paths = path.split('/')
        if len(sub_paths) == 3 and sub_paths[2][-4:] == '.obj':
            return sub_paths[1]
        else:
            return None

    dataset = ZipFileDataSet(zip_path)
    return dataset.map_keys(key_fn, inverse_key_function)
