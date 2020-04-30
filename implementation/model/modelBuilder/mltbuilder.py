import tensorflow as tf
from helper.generator.generators import nested_generator

"""Before starting refer to
https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0"""


class ModelBuilder(object):
    model_id = None
    model_parameters = None
    path_dictionary = None
    training_configuration = None

    def __init__(self, model_id, model_parameters, path_dictionary,
                 training_configuration):
        self.model_id = model_id
        self.model_parameters = model_parameters
        self.path_dictionary = path_dictionary
        self.training_configuration = training_configuration

    def get_input(self, mode):
        """Implement in derived class"""
        raise NotImplementedError()

    def get_train_inputs(self):
        return self.get_input(tf.estimator.ModeKeys.TRAIN)

    def get_predict_inputs(self):
        return self.get_input(tf.estimator.ModeKeys.PREDICT)

    def base_model(self, features, mode):
        """Implemented in the derived class
        """
        raise NotImplemented()

    def get_deformation_loss(self, predictions, ground_truth):
        """Implemented in derived class"""
        raise NotImplemented()

    def get_selection_loss(self, predictions, ground_truth):
        """Implemented in derived class"""
        raise NotImplemented()

    def get_train_op(self, loss, step):
        raise NotImplemented()

    def get_total_loss(self, inference_loss):
        """Regularization loss collected during
        training process
        Reference https://github.com/cscheau/Examples/blob/master/iris_l1_l2.py
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) > 0:
            tf.summary.scalar(
                'inference_loss', inference_loss, family='sublosses')
            """Element wise addition"""
            reg_loss = tf.add_n(reg_losses)
            tf.summary.scalar('reg_loss', reg_loss, family='sublosses')
            loss = inference_loss + reg_loss
        else:
            loss = inference_loss
        return loss

    def build_estimator(self, features, labels, mode, config=None):
        """This method will build the estimator"""
        predictions = self.base_model(features, mode)
        network_arguments = dict(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            """No need for other matrices.
            This is prediction mode"""
            return tf.estimator.EstimatorSpec(**network_arguments)

        prediction_loss = self.get_deformation_loss(predictions, labels)
        deformation_loss = self.get_total_loss(prediction_loss)
        selection_loss = self.get_selection_loss(predictions, labels)

        network_arguments['loss'] = deformation_loss + selection_loss


        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(**network_arguments)

        """Will provide current step. In this case the epoch"""
        step = tf.train.get_or_create_global_step()
        """Collection of operations. These are the operations
        that are to be executed after training step"""
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = self.get_train_op(deformation_loss=deformation_loss,
                                         selection_loss=selection_loss,
                                         step=step)

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

    def evaluate(self, config=None, **train_kwargs):
        """Wrapper around `tf.estimator.Estimator.train`."""
        estimator = self.get_estimator(config=config)
        estimator.evaluate(self.get_predict_inputs, **train_kwargs)

    def visualize_input(self, mode=tf.estimator.ModeKeys.TRAIN):
        with tf.Graph().as_default():
            if mode == tf.estimator.ModeKeys.PREDICT:
                features, targets = self.get_predict_inputs()
            elif mode == tf.estimator.ModeKeys.TRAIN:
                features, targets = self.get_train_inputs()
            with tf.train.MonitoredSession() as sess:
                while not sess.should_stop():
                    data = sess.run([features, targets])
                    for record in nested_generator(data):
                        self.visualize_data_pointcloud(*record)

    def visualize_predictions(self, mode=tf.estimator.ModeKeys.TRAIN):
        with tf.Graph().as_default():
            if mode == tf.estimator.ModeKeys.PREDICT:
                print("Visualizing test set")
                features, targets = self.get_predict_inputs()
            elif mode == tf.estimator.ModeKeys.TRAIN:
                print("Visualizing train set")
                features, targets = self.get_train_inputs()
            predictions = self.build_estimator(
                                    features, targets,
                                    tf.estimator.ModeKeys.PREDICT).predictions
            data_tensors = [predictions, features]
            if targets is not None:
                data_tensors.append(targets)
            saver = tf.train.Saver()
            with tf.train.MonitoredSession() as sess:
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.model_dir))
                while not sess.should_stop():
                    data = sess.run(data_tensors)
                    for record in nested_generator(data):
                        self.visualize_predictions_mesh(*record)

    def report_chamfer_presampled(self):
        raise NotImplemented()

    def visualize_predicted_pointclouds(self):
        raise NotImplemented()
