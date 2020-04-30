import tensorflow as tf
import random
import numpy as np
import random

from datasetReader.compressed.renderedImageReader import ImageSetReader
from datasetReader.h5py.cloudReader import CloudReader
from graphicUtils.image.utils import image_to_numpy
from graphicUtils.pointCloud.utils import sample_points

from helper.io.image import load_resized_image_from_file

def get_image_dataset(base_path, cat_id, view_angles):
    imageReader = ImageSetReader(base_path)
    imageDataset = None
    metaData = None

    """Single value"""
    if isinstance(view_angles, int):
        imageDataset, metaData = imageReader.get_single_view_dataset(
                                             cat_id, view_angles)
    elif isinstance(view_angles, (list, tuple)):
        imageDataset, metaData = imageReader.get_multi_view_dataset(
                                             cat_id, view_angles)
    else:
        raise TypeError('Invalid view angles values')
    return imageDataset.map(lambda x: image_to_numpy(x, 255)), metaData

def get_depth_image_keymap_dataset(base_path, cat_id):
    imageReader = ImageSetReader(base_path)
    imageDataset = None
    metaData = None

    imageDataset, metaData = imageReader.get_single_view_depth_dataset_keymap(cat_id)
    return imageDataset.map(lambda x: x), metaData

def get_image_keymap_dataset(base_path, cat_id):
    imageReader = ImageSetReader(base_path)
    imageDataset = None
    metaData = None

    imageDataset, metaData = imageReader.get_single_view_dataset_keymap(cat_id)
    return imageDataset.map(lambda x: image_to_numpy(x, 255)), metaData

def get_point_clouds(base_path, cat_id, n_resamples):
    cloudReader = CloudReader(base_path)
    cloudDataset = cloudReader.get_dataset(cat_id, "pointCloud")
    return cloudDataset.map(
             lambda x: sample_points(np.array(x, dtype=np.float32),
                                     n_resamples))


def get_raw_image_dataset(path_list, resolution=(192, 256)):

    def map_data(path):
        image = load_resized_image_from_file(path, resolution)
        return np.array(image)

    def map_image(path):
        image = tf.py_func(map_data, [path], (tf.uint8),
                                  stateful=False)

        image.set_shape(tuple(resolution) + (3,))
        image = tf.image.per_image_standardization(image)
        return dict(image=image, item_id=0)

    dataset = tf.data.Dataset.from_tensor_slices(
                             tf.convert_to_tensor(path_list, tf.string))


    dataset = dataset.map(map_image)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)
    return dataset


def get_dataset(base_path, cat_id, view_angles, n_resamples,
                    image_ids, shuffle=True,
                repeat=False, parallel_calls=8, batch_size=None):

    imageDataset, imageMetaData = get_image_dataset(base_path, cat_id,
                                                    view_angles)
    imageDataset.open()
    if not all(image in imageDataset for image in image_ids):
        raise KeyError("Not all images are processed")

    cloudDataset = get_point_clouds(base_path, cat_id, n_resamples)
    cloudDataset.open()
    if not all(image in cloudDataset for image in image_ids):
        raise KeyError("Not all point clouds are processed")

    dataset = tf.data.Dataset.from_tensor_slices(
                             tf.convert_to_tensor(image_ids, tf.string))

    if shuffle:
        """Sample every thing"""
        dataset = dataset.shuffle(buffer_size=len(image_ids))
    if repeat:
        dataset = dataset.repeat()

    def map_data(image_id):
        return imageDataset[image_id], cloudDataset[image_id]

    def map_image(image_id):
        """Wrap python function as tensorflow function"""
        image, cloud = tf.py_func(map_data, [image_id], (tf.uint8, tf.float32),
                                  stateful=False)

        image.set_shape(tuple((imageMetaData['shapeX'],
                               imageMetaData['shapeY'])) + (3,))
        cloud.set_shape((n_resamples, 3))

        image = tf.image.per_image_standardization(image)
        return dict(item_id=image_id, image=image), cloud

    dataset = dataset.map(map_image, num_parallel_calls=parallel_calls)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    """Prefetch first two datasets"""
    dataset = dataset.prefetch(2)
    return dataset


def get_depth_dataset(base_path, cat_id, view_angles, image_ids, shuffle=True,
                        repeat=False, parallel_calls=8, batch_size=None,
                        target_height=55, target_width=74):

    imageDataset, imageDatasetMetadata = get_image_keymap_dataset(base_path,
                                                                  cat_id)

    imageDepthset, imageDepthsetMetadata = get_depth_image_keymap_dataset(base_path,
                                                                          cat_id)

    imageDataset.open()
    imageDepthset.open()

    dataset = tf.data.Dataset.from_tensor_slices(
                             tf.convert_to_tensor(image_ids, tf.string))

    if shuffle:
        """Sample every thing"""
        dataset = dataset.shuffle(buffer_size=len(image_ids))
    if repeat:
        dataset = dataset.repeat()

    def map_data(image_id):
        angle = random.sample(view_angles, 1)[0]
        return imageDataset[(image_id, angle)], imageDepthset[(image_id, angle)]

    def map_image(image_id):
        """Wrap python function as tensorflow function"""
        image, depth = tf.py_func(map_data, [image_id], (tf.uint8, tf.string),
                                  stateful=False)

        depth = tf.image.decode_png(depth, channels=1)
        image.set_shape(tuple((imageDatasetMetadata['shapeX'],
                               imageDatasetMetadata['shapeY'])) + (3,))

        depth.set_shape(tuple((imageDatasetMetadata['shapeX'],
                               imageDatasetMetadata['shapeY'], 1)))

        depth = tf.cast(depth, tf.float32)
        depth = tf.div(depth, [255.0])
        depth = tf.image.resize_images(depth, (target_height, target_width))
        image = tf.image.per_image_standardization(image)
        invalid_depth = tf.sign(depth)
        return dict(image_id=image_id, image=image), dict(depth=depth,
                                                          invalid_depth=invalid_depth)


    dataset = dataset.map(map_image, num_parallel_calls=parallel_calls)

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    """Prefetch first two datasets"""
    dataset = dataset.prefetch(2)
    return dataset
