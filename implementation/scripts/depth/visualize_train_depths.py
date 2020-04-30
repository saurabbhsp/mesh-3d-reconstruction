import tensorflow as tf
import argparse
import configparser
from model.depthEstimator.model import DepthEstimator
from configReader import depth_config_reader as configReader

"""
This script initialized the training process. For training process
a single parameter specifying the path to configuration file is required.
"""

parser = argparse.ArgumentParser(description="Train FFD model")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.configFile)

"""Read config"""
category = configReader.get_category(config)
path_dictionary = configReader.get_path_list(config)

split_config = configReader.get_split_params(config)
run_config = tf.estimator.RunConfig(**configReader.get_run_config(config))

id = configReader.get_depth_model_id(config)
max_steps = configReader.get_max_steps(config)

train_config = configReader.get_train_config(config)

model = DepthEstimator(id, category,
                           path_dictionary, split_config, train_config)
model.visualize_training()
