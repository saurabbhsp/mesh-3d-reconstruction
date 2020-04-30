import os
from dictionaryDataset.zipfileDataDictionary import ZipFileDataSet
from helper.io.image import load_image_from_file, load_from_file
import json
import random

class ImageSetReader(object):

    def __init__(self, base_path):
        self._base_path = base_path

    def get_relative_rendered_image(self, item_id, angle):
        if type(item_id) == bytes:
            item_id = item_id.decode("utf-8")
        return os.path.join(item_id,
                            '%s_r_%03d.png' % (item_id, angle))

    def get_relative_depth_image(self, item_id, angle):
        if type(item_id) == bytes:
            item_id = item_id.decode("utf-8")
        return os.path.join(item_id,
                            '%s_r_%03d_depth.png0001.png' % (item_id, angle))

    def get_zip_path(self, cat_id):
        return os.path.join(self._base_path, "images", '%s.zip' % cat_id)

    def get_multi_view_dataset(self, cat_id, angles):

        def key_fn(key):
            angle = random.sample(angles, 1)[0]
            return self.get_relative_rendered_image(key, angle)
        metadata = ZipFileDataSet(self.get_zip_path(cat_id))
        metadata.open()
        _metadata = json.loads(metadata['metadata.json'].
                               readline().decode('utf-8'))
        metadata.close()

        dataset = ZipFileDataSet(self.get_zip_path(cat_id))
        dataset = dataset.map(load_image_from_file)
        dataset = dataset.map_keys(key_fn)
        return dataset, _metadata

    def get_single_view_dataset(self, cat_id, angle):

        def key_fn(key):
            return self.get_relative_rendered_image(key, angle)

        metadata = ZipFileDataSet(self.get_zip_path(cat_id))
        metadata.open()
        _metadata = json.loads(metadata['metadata.json'].
                               readline().decode('utf-8'))
        metadata.close()

        dataset = ZipFileDataSet(self.get_zip_path(cat_id))
        dataset = dataset.map(load_image_from_file)
        dataset = dataset.map_keys(key_fn)
        return dataset, _metadata


    def get_single_view_depth_dataset_keymap(self, cat_id):

        def key_fn(key):
            return self.get_relative_depth_image(key[0], key[1])

        metadata = ZipFileDataSet(self.get_zip_path(cat_id))
        metadata.open()
        _metadata = json.loads(metadata['metadata.json'].
                               readline().decode('utf-8'))
        metadata.close()

        dataset = ZipFileDataSet(self.get_zip_path(cat_id))
        dataset = dataset.map(load_from_file)
        dataset = dataset.map_keys(key_fn)
        return dataset, _metadata

    def get_single_view_dataset_keymap(self, cat_id):

        def key_fn(key):
            return self.get_relative_rendered_image(key[0], key[1])

        metadata = ZipFileDataSet(self.get_zip_path(cat_id))
        metadata.open()
        _metadata = json.loads(metadata['metadata.json'].
                               readline().decode('utf-8'))
        metadata.close()

        dataset = ZipFileDataSet(self.get_zip_path(cat_id))
        dataset = dataset.map(load_image_from_file)
        dataset = dataset.map_keys(key_fn)
        return dataset, _metadata
