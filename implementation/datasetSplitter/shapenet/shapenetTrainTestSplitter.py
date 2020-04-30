import random
import os
import tensorflow as tf

from helper.shapenet.datareader.reader import DataReader
from templateManager.shapenet.shapenetTemplateManager import get_template_ids
from helper.io.serialize import read_list, write_list


def _split(data, train_percentage):
    size = len(data)
    train_set = data[0:int(size * train_percentage)]
    test_set = data[int(size * (train_percentage)):]
    return train_set, test_set


class Splitter(object):

    output_path = None
    cat_id = None
    template_ids = None
    train_set = None
    test_set = None

    def __init__(self, output_path, input_path, cat_id, train_split=0.8,
                 random_seed=0, shuffle=True,
                 replace=True):

        self.output_path = output_path
        self.cat_id = cat_id
        random.seed(random_seed)
        self.template_ids = set(get_template_ids(self.cat_id))

        base_path = os.path.join(output_path, cat_id)
        os.makedirs(base_path, exist_ok=True)
        if replace or not os.path.exists(os.path.join(base_path,
                                                  "train.pkl")):
            data_reader = DataReader(input_path)
            data_ids = data_reader.list_archived_data(self.cat_id)
            data_ids = [data_id for data_id in data_ids if data_id
                        not in self.template_ids]
            data_ids.sort()
            if shuffle:
                random.shuffle(data_ids)

            train, test = _split(data_ids, train_split)
            self.train_set = train
            self.test_set = test

            write_list(os.path.join(base_path, "train.pkl"), train)
            write_list(os.path.join(base_path, "test.pkl"), test)
        else:
            self.train_set = read_list(os.path.join(base_path,
                                       "train.pkl"))
            self.test_set = read_list(os.path.join(base_path,
                                                   "test.pkl"))

    def get_data(self, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.train_set
        elif tf.estimator.ModeKeys.PREDICT:
            return self.test_set
        else:
            raise ValueError("No such mode exists")
