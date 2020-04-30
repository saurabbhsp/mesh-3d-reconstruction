import os
import tensorflow as tf
import numpy as np
from keras import backend as K

from model.modelBuilder.builder import ModelBuilder
from helper.shapenet.shapenetMapper import desc_to_id
from datasetSplitter.shapenet.shapenetTrainTestSplitter import Splitter
from datasetReader.reader import get_dataset, get_point_clouds

from keras.applications.mobilenet import MobileNet
from templateManager.shapenet.shapenetTemplateManager import get_template_ids
from deformations.FFD import get_template_ffd
from deformations.meshDeformation import get_thresholded_template_mesh
from datasetReader.h5py.meshReader import MeshReader

from metrics.chamfer.chamfer import bidirectionalchamfer
from metrics.chamfer.normalizedChamfer import get_normalized_chamfer

import matplotlib.pyplot as plt
from graphicUtils.visualizer.mayaviVisualizer import visualize_point_cloud
from graphicUtils.visualizer.mayaviVisualizer import visualize_mesh
from graphicUtils.pointCloud.utils import sample_points
from mayavi import mlab
from helper.generator.generators import nested_generator


def add_update_ops(ops):
    """
    Add specified ops to UPDATE_OPS collection if not already present.

    Newer versions of tf.keras.Model add update ops to the models update s,
    but not to the graph collection. This fixes that if they are expected to
    also be added to tf.GraphKeys.UPDATE_OPS.

    Args:
        ops: iterable of operations to be added to tf.GraphKeys.UPDATE_OPS if
            not already present
    Returns:
        None
    """
    update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    ops = set(ops)
    for op in ops:
        if op not in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, op)


def initialize_uninitialized_variables(sess):
    global_vars = tf.global_variables()
    is_init = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    init_vars = [v for (v, i) in zip(global_vars, is_init) if not i]
    sess.run(tf.variables_initializer(init_vars))


def batch_normalized_activation(activation, **normalization_arguments):
    def activated_function(input):
        return activation(tf.layers.batch_normalization(
                                                input,
                                                **normalization_arguments))
    return activated_function


def exp_annealing_factor(rate):
    step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
    return tf.exp(-step*rate)


def linear_annealing_factor(cutoff):
    step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
    return tf.maximum(1 - step / cutoff, 0)


def annealed_weight(weight, linear_annealing_cutoff=None,
                    exp_annealing_rate=None):
    if linear_annealing_cutoff is None:
        if exp_annealing_rate is None:
            return weight
        else:
            return weight * exp_annealing_factor(exp_annealing_rate)
    else:
        if exp_annealing_rate is None:
            return weight*linear_annealing_factor(linear_annealing_cutoff)
        else:
            raise ValueError("")


def sample_tf(x, n_resamples, axis=0, name=None):
    n_original = x.shape[axis]
    indices = tf.random_uniform(
        shape=(n_resamples,), minval=0, maxval=n_original, dtype=np.int32)
    return tf.gather(x, indices, axis=axis, name=name)


class TemplateFFDBuilder(ModelBuilder):

    def __init__(self, model_id, model_parameters,
                 path_dictionary, split_params, training_configuration):
        super(TemplateFFDBuilder, self).__init__(model_id,
                                                 model_parameters,
                                                 path_dictionary,
                                                 training_configuration)
        self.data_splitter = Splitter(self.path_dictionary["output_path"],
                                      self.path_dictionary["input_path"],
                                      self.cat_id, **split_params)
        self._model_initialized = True

    def get_dataset(self, mode):
        dataset_ids = self.data_splitter.get_data(mode)
        """For testing purpose
        dataset_ids = [ '10640377f4eb9ecdadceecd3bc8bde14', '115aa37af1a07d24a5a88312547ed863', '1a640c8dffc5d01b8fd30d65663cfd42',
                       '2091ab9f69c77273de2426af5ed9b6a', '2eca5fa283b317c7602717bb378973f1', '383ed236166980209e23d6b6aa619041',
                       '3a8478c9f2c1c043eb81825856d1297f', '425abc480a0b390d7cc46b39c0cc084b',
                       '42de9b896d23244fe6fbd395d87e5106', '446e4145b475eb245751d640a4e334']"""

        dataset = get_dataset(self.preprocessed_data_path,
                              self.cat_id,
                              self.view_angles,
                              self.cloud_resamples,
                              dataset_ids,
                              mode == tf.estimator.ModeKeys.TRAIN,
                              mode == tf.estimator.ModeKeys.TRAIN,
                              batch_size=self.batch_size
                              )
        return dataset

    def get_input(self, mode):
        dataset = self.get_dataset(mode)
        return dataset.make_one_shot_iterator().get_next()


    def initialize_execution_graph(self):
        """Load training data"""
        if os.path.exists(self.model_dir) and len(
                                            os.listdir(self.model_dir)) > 0:
            self._model_initialized = True
            print("Execution graph already initialized")
            return
        os.makedirs(self.model_dir,
                    exist_ok=True)
        self._model_initialized = False
        try:
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    features, labels = self.get_train_inputs()
                    self.build_estimator(features, labels,
                                         tf.estimator.ModeKeys.TRAIN)
                    initialize_uninitialized_variables(sess)
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(self.model_dir,
                               self.model_id), global_step=0)
        except Exception:
            self._model_initialized = False
            raise
        self._model_initialized = True

    def get_image_features(self, image, mode, **inference_params):
        alpha = inference_params.get('alpha', 1)
        """Might need some changes"""
        if self._model_initialized:
            weights = None
        else:
            weights = 'imagenet'
        isTraining = mode == tf.estimator.ModeKeys.TRAIN
        K.set_learning_phase(isTraining)
        """For mobile net v2 Casting is needed. Skipped
        for time being. Will be added later."""
        model = MobileNet(input_shape=image.shape.as_list()[1:],
                          input_tensor=image,
                          include_top=False,
                          weights=weights,
                          alpha=alpha)
        add_update_ops(model.updates)
        features = model.output
        conv_filters = inference_params.get("image_conv_filters", [64])

        for filter in conv_filters:
            features = tf.layers.conv2d(features, filter, 1)
            features = tf.nn.relu6(tf.layers.batch_normalization(features))

        return features

    def base_model(self, features, mode):
        inference_params = self.model_parameters.get("inference_params", {})
        isTraining = mode == tf.estimator.ModeKeys.TRAIN

        image = features['image']
        item_id = features['item_id']
        """Mobile net"""
        features = self.get_image_features(image, mode,
                                           **inference_params)
        features = tf.layers.flatten(features)
        """Shared FC"""
        dense_layers = inference_params.get("dense_layers", [512])
        for dense_layer in dense_layers:
            features = tf.layers.dense(features, dense_layer,
                                       activation=batch_normalized_activation(
                                        tf.nn.relu6, training=isTraining)
                                       )
        """Template FC"""
        deformed_points = tf.layers.dense(
                    features,
                    self.template_count * self.n_control_points * 3,
                    kernel_initializer=tf.random_normal_initializer(
                                                                   stddev=1e-4)
                             )
        """Reshape for each template"""
        deformed_points = tf.reshape(deformed_points, (-1, self.template_count,
                                     self.n_control_points, 3))
        probs = tf.layers.dense(
            features, self.template_count, activation=tf.nn.softmax)
        """Gamma calculation. This increases deformation by +1"""
        eps = self.model_parameters.get('prob_eps', 0.1)
        if eps > 0:
            gamma = (1 - eps)*probs + eps / self.template_count

        return dict(item_id=item_id,
                    probs=gamma,
                    deformed_points=deformed_points)

    def get_train_op(self, loss, step):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.model_parameters.get('learning_rate', 1e-3))
        return optimizer.minimize(loss, step)

    def get_loss(self, predictions, ground_truth):
        losses = []
        probs, deformed_points = (predictions[i] for i in ('probs',
                                                           'deformed_points'))
        gamma_mode = self.model_parameters.get('gamma', 'linear')
        gamma = self._calculate_gamma_param(probs, gamma_mode)

        chamfer_loss = self.get_chamfer_loss(gamma, deformed_points,
                                             ground_truth)
        tf.summary.scalar('chamfer', chamfer_loss, family='sublosses')
        losses.append(chamfer_loss)

        entropy_loss_params = self.model_parameters.get('entropy_loss')
        if entropy_loss_params is not None:
            print("Entropy loss called")
            entropy_loss = self.get_entropy_loss(probs, **entropy_loss_params)
            tf.summary.scalar('entropy_loss', entropy_loss, family='sublosses')
            losses.append(entropy_loss)

        deformation_regularization_params = self.model_parameters.get(
                                                'dp_regularization')
        if deformation_regularization_params is not None:
            regularization = self.get_deformation_regularization(
                                        probs, deformed_points,
                                        **deformation_regularization_params
                                                                 )
            tf.summary.scalar('deformation_regularization', regularization,
                              family='sublosses')
            losses.append(regularization)
        if len(losses) == 1:
            return losses[0]
        else:
            return tf.add_n(losses)

    """Non linear weighting penalty"""
    def _calculate_gamma_param(self, gamma, gamma_mode):
        if gamma_mode == 'linear':
            return gamma
        elif gamma_mode == 'square':
            return gamma ** 2
        elif gamma_mode == 'log':
            return -tf.log(1 - gamma)
        else:
            raise ValueError("Invalid Gamma Mode")

    """All required losses"""
    def get_chamfer_loss(self, gamma, deformed_points, ground_truth):
        inferred_point_clouds = self.get_inferred_point_clound(deformed_points)
        inferred_point_clouds = tf.unstack(inferred_point_clouds, axis=1)
        losses = [bidirectionalchamfer(inferred, ground_truth)
                  for inferred in inferred_point_clouds]
        losses = tf.stack(losses, axis=1)
        losses = gamma * losses
        loss = tf.reduce_sum(losses)
        return loss

    def get_entropy_loss(self, probs, **weight_kwargs):
        mean_probs = tf.reduce_mean(probs, axis=0)
        entropy_loss = tf.reduce_sum(mean_probs * tf.log(mean_probs))
        weight = annealed_weight(**weight_kwargs)
        return entropy_loss * weight

    def get_deformation_regularization(self, probs, deformed_points,
                                       **weight_kwargs):
        if weight_kwargs.pop('uniform', False):
            reg_loss = tf.reduce_sum(deformed_points**2)
        else:
            reg_loss = tf.reduce_sum(deformed_points**2, axis=(2, 3))
            reg_loss *= probs
            reg_loss = tf.reduce_sum(reg_loss)
        weight = annealed_weight(**weight_kwargs)
        return reg_loss * weight

    def get_inferred_point_clound(self, deformed_points):
        b, p = self.get_ffd_tensors()
        inferred_point_clouds = tf.einsum('ijk,likm->lijm', b,
                                          p + deformed_points)
        return inferred_point_clouds

    """Need to check the reason for resampling"""
    def get_ffd_tensors(self, ffd_dataset=None):
        bs = []
        ps = []

        """b is the deformation matrix and p are
        the points"""
        for template, b, p in self.get_ffd_data(ffd_dataset):
            b = sample_tf(tf.constant(b, dtype=tf.float32),
                          self.n_ffd_resamples,  axis=0,
                          name='b_resampled_'+str(template))
            bs.append(b)
            ps.append(p)
        b = tf.stack(bs)
        p = tf.constant(np.array(ps), dtype=tf.float32)
        return b, p

    """We want to return tuple hence seperate method"""
    def _get_ffd_data(self, ffd_dataset):
        for template in self.template_ids:
            data = ffd_dataset[template]
            b, p = (np.array(data[k]) for k in ('b', 'p'))
            yield template, b, p

    def get_ffd_data(self, ffd_dataset=None):
        if ffd_dataset is None:
            n_ffd_points = self.n_ffd_samples
            ffd_dataset = get_template_ffd(self.preprocessed_data_path,
                                           self.cat_id,
                                           n_samples=n_ffd_points)
        with ffd_dataset:
            return tuple(self._get_ffd_data(ffd_dataset))

    def _mesh_transformation(self, edge_length_threshold=0.02):
        ffd_dataset = get_template_ffd(
                                self.preprocessed_data_path,
                                self.cat_id,
                                edge_length_threshold)

        template_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))
        mesh_dataset = get_thresholded_template_mesh(
                                    self.preprocessed_data_path,
                                    self.cat_id, edge_length_threshold)
        with mesh_dataset:
            all_faces = []
            all_vertices = []
            for k in template_ids:
                sg = mesh_dataset[k]
                all_faces.append(np.array(sg['faces']))
                all_vertices.append(np.array(sg['vertices']))

        def transform_predictions(probs, dp):
            i = np.argmax(probs)
            vertices = np.matmul(bs[i], ps[i] + dp[i])
            faces = all_faces[i]
            original_vertices = all_vertices[i]
            return dict(
                vertices=vertices,
                faces=faces,
                original_vertices=original_vertices,
                attrs=dict(template_id=template_ids[i]))
        return transform_predictions

    def visualize_data_pointcloud(self, feature_data, target):
        image = feature_data['image']
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        plt.show()
        visualize_point_cloud(target, color=(1, 0, 0), scale_factor=0.01)
        mlab.show()
        plt.close()

    def visualize_predictions_mesh(self, prediction_data,
                                   feature_data, target = None):

        image = feature_data['image']
        dp = prediction_data['deformed_points']
        probs = prediction_data['probs']

        if not hasattr(self, '_mesh_fn') or self._mesh_fn is None:
            self._mesh_fn = self._mesh_transformation()

        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        plt.show()

        mesh = self._mesh_fn(probs, dp)
        vertices, faces, original_vertices = (
            mesh[k] for k in('vertices', 'faces', 'original_vertices'))

        visualize_mesh(original_vertices, faces)
        mlab.show()

        visualize_mesh(vertices, faces, color=(1, 0, 0),
                       include_wireframe=False)
        mlab.show()

        visualize_point_cloud(vertices, color=(1, 0, 0), scale_factor=0.01)
        mlab.show()

    def report_chamfer_presampled(self):
        evaluation_ids = self.data_splitter.get_data(tf.estimator.ModeKeys.PREDICT)

        point_cloud_dataset = get_point_clouds(self.preprocessed_data_path,
                                               self.cat_id, self.n_ffd_resamples)
        point_cloud_dataset = point_cloud_dataset.subset(evaluation_ids)

        mesh_dataset = MeshReader(self.preprocessed_data_path).get_dataset(self.cat_id)
        mesh_dataset = mesh_dataset.subset(evaluation_ids)
        mesh_dataset.open()

        deformed_predictions = []
        ground_truth_point_cloud = []
        mesh_ground_truth = []

        ffd_dataset = get_template_ffd(
                                self.preprocessed_data_path,
                                self.cat_id,
                                edge_length_threshold=None)

        template_ids, bs, ps = zip(*self.get_ffd_data(ffd_dataset))

        with tf.Graph().as_default():
            dataset = get_dataset(self.preprocessed_data_path,
                                  self.cat_id,
                                  self.view_angles,
                                  self.cloud_resamples,
                                  evaluation_ids,
                                  False,
                                  False,
                                  batch_size=len(evaluation_ids)
                                  )
            features, targets = dataset.make_one_shot_iterator().get_next()
            predictions = self.build_estimator(
                                features, targets,
                                tf.estimator.ModeKeys.PREDICT).predictions
            saver = tf.train.Saver()
            with tf.train.MonitoredSession() as sess:
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.model_dir))
                data = sess.run(predictions)
                point_cloud_dataset.open()
                for evaluation_id, prediction_tensor in zip(evaluation_ids, nested_generator(data)):
                    dp = prediction_tensor['deformed_points']
                    probs = prediction_tensor['probs']
                    i = np.argmax(probs)
                    predicted_vertices = np.matmul(bs[i], ps[i] + dp[i])
                    deformed_predictions.append(sample_points(predicted_vertices, self.n_ffd_resamples))
                    ground_truth_point_cloud.append(point_cloud_dataset[evaluation_id])
                    mesh_ground_truth.append(mesh_dataset[evaluation_id])
            chamfer_list, unnorm_chamfer = get_normalized_chamfer(mesh_ground_truth, ground_truth_point_cloud,
                                                  deformed_predictions, self.n_ffd_resamples)
            print("The normalized chamfer for test set is "+str(np.mean(chamfer_list)))
            print("The non normalized chamfer for test set is "+str(np.mean(unnorm_chamfer)))


    def visualize_predicted_pointclouds(self):
        evaluation_ids = self.data_splitter.get_data(tf.estimator.ModeKeys.PREDICT)

        template_ids, bs, ps = zip(*self.get_ffd_data())

        with tf.Graph().as_default():
            dataset = get_dataset(self.preprocessed_data_path,
                                  self.cat_id,
                                  self.view_angles,
                                  self.cloud_resamples,
                                  evaluation_ids,
                                  False,
                                  False,
                                  batch_size=len(evaluation_ids)
                                  )
            features, targets = dataset.make_one_shot_iterator().get_next()
            predictions = self.build_estimator(
                                features, targets,
                                tf.estimator.ModeKeys.PREDICT).predictions
            saver = tf.train.Saver()
            with tf.train.MonitoredSession() as sess:
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.model_dir))
                data = sess.run([features, predictions])
                for prediction_tensor in nested_generator(data):
                    image = prediction_tensor[0]['image']
                    image -= np.min(image)
                    image /= np.max(image)
                    plt.imshow(image)
                    plt.show()

                    dp = prediction_tensor[1]['deformed_points']
                    probs = prediction_tensor[1]['probs']
                    i = np.argmax(probs)
                    predicted_vertices = np.matmul(bs[i], ps[i] + dp[i])

                    visualize_point_cloud(np.matmul(bs[i], ps[i]), color=(0, 0, 1), scale_factor=0.01)
                    mlab.show()

                    visualize_point_cloud(predicted_vertices, color=(1, 0, 0), scale_factor=0.01)
                    mlab.show()

    @property
    def cat_id(self):
        return desc_to_id(self.model_parameters["cat_desc"])

    @property
    def n_ffd_samples(self):
        return self.training_configuration.get('n_ffd_samples', 16384)

    @property
    def preprocessed_data_path(self):
        return self.path_dictionary['preprocessed_data_path']

    @property
    def view_angles(self):
        return self.training_configuration.get("views", 225)

    @property
    def cloud_resamples(self):
        return self.training_configuration.get("point_cloud_sub_samples", 1024)

    @property
    def batch_size(self):
        return self.training_configuration.get("batch_size", 32)

    @property
    def output_path(self):
        return self.path_dictionary["output_path"]

    @property
    def model_dir(self):
        return os.path.join(self.output_path, "model",
                            self.model_parameters["cat_desc"],
                            self.model_id)

    @property
    def slices(self):
        return self.training_configuration.get('slices', 3)

    @property
    def n_control_points(self):
        """Each slice will create n+1 partitions
        and in 3D geometry these partitions will
        result into (n+1)**3 control points"""
        return (self.slices + 1) ** 3

    @property
    def template_ids(self):
        return get_template_ids(self.cat_id)

    @property
    def template_count(self):
        return len(self.template_ids)


    @property
    def n_ffd_resamples(self):
        return self.training_configuration.get('n_ffd_resamples', 1024)
