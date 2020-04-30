import tensorflow as tf
import os
import math
import numpy as np

import matplotlib.pyplot as plt
from datasetSplitter.shapenet.shapenetTrainTestSplitter import Splitter
from datasetReader.reader import get_depth_dataset
from helper.shapenet.shapenetMapper import desc_to_id
from helper.generator.generators import nested_generator
from PIL import Image


def initialize_uninitialized_variables(sess):
    global_vars = tf.global_variables()
    is_init = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    init_vars = [v for (v, i) in zip(global_vars, is_init) if not i]
    sess.run(tf.variables_initializer(init_vars))

def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_gpu(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv_ = tf.nn.relu(bias, name=scope.name)
        return conv_

def fc(scope_name, inputs, shape, bias_shape, wd=0.04, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse is True:
            scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
        fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)
        return fc



class DepthEstimator(object):

    model_id = None
    category = None
    path_dictionary = None
    training_configuration = None


    def __init__(self, model_id, category,
                 path_dictionary, split_params, training_configuration):

        self.model_id = model_id
        self.path_dictionary = path_dictionary
        self.category = category
        self.training_configuration = training_configuration

        if split_params is not None:
            self.data_splitter = Splitter(self.output_path,
                                      self.input_path,
                                      self.cat_id, **split_params)
        else:
            self.data_splitter = None

        if os.path.exists(self.model_dir) and len(
                                            os.listdir(self.model_dir)) > 0:
            print("Depth Model is already initialized")
            return
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    features, targets = self.get_train_inputs()
                    self.build_estimator(features, targets,
                                         tf.estimator.ModeKeys.TRAIN)
                    initialize_uninitialized_variables(sess)
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(self.model_dir,
                               self.model_id), global_step=0)

    def get_dataset(self, mode):
        dataset_ids = self.data_splitter.get_data(mode)
        self.examples_count = len(dataset_ids)
        print(self.examples_count)
        dataset = get_depth_dataset(self.preprocessed_data_path, self.cat_id,
                                    self.view_angles, dataset_ids,
                                    mode == tf.estimator.ModeKeys.TRAIN,
                                    mode == tf.estimator.ModeKeys.TRAIN,
                                    batch_size=self.batch_size,
                                    target_height=self.target_height,
                                    target_width=self.target_width)
        return dataset


    def get_input(self, mode):
        dataset = self.get_dataset(mode)
        return dataset.make_one_shot_iterator().get_next()

    def get_train_inputs(self):
        return self.get_input(tf.estimator.ModeKeys.TRAIN)

    def get_loss(self, predictions, ground_truth):
        predictions_flat = tf.reshape(predictions, [-1, self.target_width * self.target_height])
        ground_truth_flat = tf.reshape(ground_truth['depth'], [-1, self.target_width * self.target_height])
        invalid_depths_flat = tf.reshape(ground_truth['invalid_depth'], [-1, self.target_width * self.target_height])

        target = tf.multiply(ground_truth_flat, invalid_depths_flat)
        predict = tf.multiply(predictions_flat, invalid_depths_flat)
        d = tf.subtract(predict, target)
        square_d = tf.square(d)
        sum_square_d = tf.reduce_sum(square_d, 1)
        sum_d = tf.reduce_sum(d, 1)
        sqare_sum_d = tf.square(sum_d)
        cost = tf.reduce_mean(sum_square_d / self.target_width * self.target_height - 0.5 * sqare_sum_d / math.pow(self.target_width * self.target_height, 2))
        tf.add_to_collection('losses', cost)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
        return cost

    def add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def get_train_op(self, loss, step):
        decay_steps = 30 * math.ceil(self.examples_count/self.batch_size)
        lr = tf.train.exponential_decay(
            1e-4,
            step,
            decay_steps,
            0.9,
            staircase=True)
        tf.summary.scalar('learning_rate', lr)
        loss_averages_op = self.add_loss_summaries(loss)
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=step)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.999999, step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


    def base_model(features, mode, height, width):
        images = features['image']
        isTrainable = mode == tf.estimator.ModeKeys.TRAIN
        coarse1_conv = conv2d('coarse1', images, [11, 11, 3, 96], [96], [1, 4, 4, 1], padding='VALID', reuse=False, trainable=isTrainable)
        coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        coarse2_conv = conv2d('coarse2', coarse1, [5, 5, 96, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=False, trainable=isTrainable)
        coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        coarse3 = conv2d('coarse3', coarse2, [3, 3, 256, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=False, trainable=isTrainable)
        coarse4 = conv2d('coarse4', coarse3, [3, 3, 384, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=False, trainable=isTrainable)
        coarse5 = conv2d('coarse5', coarse4, [3, 3, 384, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=False, trainable=isTrainable)
        coarse6 = fc('coarse6', coarse5, [3*7*256, 5000], [5000], reuse=False, trainable=isTrainable)
        coarse7 = fc('coarse7', coarse6, [5000, 4070], [4070], reuse=False, trainable=isTrainable)

        logits = tf.reshape(coarse7, [-1, height, width, 1])
        return logits


    def build_estimator(self, features, labels, mode, config=None):
        predictions = DepthEstimator.base_model(features, mode,
                                                self.target_height, self.target_width)
        network_arguments = dict(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            """No need for other matrices.
            This is prediction mode"""
            return tf.estimator.EstimatorSpec(**network_arguments)

        print("Calculating loss")
        loss = self.get_loss(predictions, labels)
        network_arguments['loss'] = loss

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(**network_arguments)

        """Will provide current step. In this case the epoch"""
        step = tf.train.get_or_create_global_step()
        """Collection of operations. These are the operations
        that are to be executed after training step"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = self.get_train_op(loss=loss, step=step)
        network_arguments['train_op'] = train_op

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(**network_arguments)


    def get_estimator(self, config=None):
        """Get the `tf.estimator.Estimator` defined by this builder."""
        return tf.estimator.Estimator(
            self.build_estimator, self.model_dir, config=config)

    def train(self, config=None, **train_kwargs):
        """Wrapper around `tf.estimator.Estimator.train`."""
        estimator = self.get_estimator(config=config)
        estimator.train(self.get_train_inputs, **train_kwargs)


    def restore(saver, session, path):
        saver.restore(session, tf.train.latest_checkpoint(path))



    def predict(features, height, width):
        predictions = DepthEstimator.base_model(features, tf.estimator.ModeKeys.PREDICT,
                                                height, width)
        return tf.transpose(predictions, (0, 3, 1, 2))

    def visualize_predictions(self):

        with tf.Graph().as_default():

            features, targets = self.get_input(tf.estimator.ModeKeys.PREDICT)
            predictions = DepthEstimator.predict(features,
                                                 self.target_height, self.target_width)

            data_tensors = [predictions, features, targets]

            if targets is not None:
                data_tensors.append(targets)
            saver = tf.train.Saver()
            with tf.train.MonitoredSession() as sess:

                DepthEstimator.restore(saver, sess, self.model_dir)

                while not sess.should_stop():
                    data = sess.run(data_tensors)
                    for record in nested_generator(data):
                        image = record[1]['image']
                        image -= np.min(image)
                        image /= np.max(image)
                        plt.imshow(image)
                        plt.show()
                        plt.clf()
                        depth = record[0]

                        if np.max(depth) != 0:
                             ra_depth = (depth/np.max(depth)) * 255.0
                        else:
                            ra_depth = depth * 255.0

                        gtr_depth = record[2]['depth'].transpose(2, 0, 1)
                        if np.max(gtr_depth) != 0:
                             gtr_depth = (gtr_depth/np.max(gtr_depth)) * 255.0
                        else:
                            gtr_depth = gtr_depth * 255.0

                        plt.imshow(gtr_depth[0])
                        plt.show()
                        plt.clf()

                        plt.imshow(ra_depth[0])
                        plt.show()
                        plt.clf()


    def visualize_training(self):

        with tf.Graph().as_default():

            features, targets = self.get_input(tf.estimator.ModeKeys.TRAIN)
            predictions = DepthEstimator.predict(features,
                                                 self.target_height, self.target_width)
            data_tensors = [predictions, features, targets]

            if targets is not None:
                data_tensors.append(targets)
            saver = tf.train.Saver()
            with tf.train.MonitoredSession() as sess:

                DepthEstimator.restore(saver, sess, self.model_dir)

                while not sess.should_stop():
                    data = sess.run(data_tensors)
                    for record in nested_generator(data):
                        image = record[1]['image']
                        image -= np.min(image)
                        image /= np.max(image)
                        plt.imshow(image)
                        plt.show()
                        plt.clf()
                        depth = record[0]

                        if np.max(depth) != 0:
                             ra_depth = (depth/np.max(depth)) * 255.0
                        else:
                            ra_depth = depth * 255.0

                        gtr_depth = record[2]['depth'].transpose(2, 0, 1)
                        if np.max(gtr_depth) != 0:
                             gtr_depth = (gtr_depth/np.max(gtr_depth)) * 255.0
                        else:
                            gtr_depth = gtr_depth * 255.0

                        plt.imshow(gtr_depth[0])
                        plt.show()
                        plt.clf()

                        plt.imshow(ra_depth[0])
                        plt.show()
                        plt.clf()




    """Properties"""
    @property
    def output_path(self):
        return self.path_dictionary["output_path"]

    @property
    def input_path(self):
        return self.path_dictionary["input_path"]

    @property
    def batch_size(self):
        return self.training_configuration.get("batch_size", 32)

    @property
    def view_angles(self):
        return self.training_configuration.get("views", 225)

    @property
    def preprocessed_data_path(self):
        return self.path_dictionary['preprocessed_data_path']

    @property
    def cat_id(self):
        return desc_to_id(self.category)

    @property
    def target_height(self):
        return self.training_configuration.get("target_height", 55)

    @property
    def target_width(self):
        return self.training_configuration.get("target_width", 74)

    @property
    def model_dir(self):
        return os.path.join(self.output_path, "model", "depth",
                            self.category,
                            self.model_id)
