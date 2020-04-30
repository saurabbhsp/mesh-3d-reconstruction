import tensorflow as tf
from metrics.chamfer.nearest_neighbour_cuda import  nn_distance as nn_distance_cpu


def bidirectionalchamfer(pointCloud1, pointCloud2):

    with tf.name_scope('bidirectionalchamfer'):
        shape1 = pointCloud1.shape.as_list()
        shape2 = pointCloud2.shape.as_list()

        pointCloud1 = tf.reshape(pointCloud1, [-1] + shape1[-2:])
        pointCloud2 = tf.reshape(pointCloud2, [-1] + shape2[-2:])

        dist1, _, dist2, __ = nn_distance_cpu(pointCloud1, pointCloud2)
        loss1 = tf.reduce_sum(dist1, axis=-1)
        loss2 = tf.reduce_sum(dist2, axis=-1)

        if len(shape1) > 3:
                loss1 = tf.reshape(loss1, shape1[:-2])
        if len(shape2) > 3:
                loss2 = tf.reshape(loss2, shape2[:-2])

        return loss1 + loss2
