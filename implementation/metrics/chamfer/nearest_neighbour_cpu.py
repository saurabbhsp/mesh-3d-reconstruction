import tensorflow as tf


def nn_distance_cpu(pc1, pc2):
    '''
    Input:
        pc1: float TF tensor in shape (B,N,C) the first point cloud
        pc2: float TF tensor in shape (B,M,C) the second point cloud
    Output:
        dist1: float TF tensor in shape (B,N) distance from first to second
        idx1: int32 TF tensor in shape (
        B,N) nearest neighbor from first
        to second
        dist2: float TF tensor in shape (B,M) distance from second to first
        idx2: int32 TF tensor in shape (B,M) nearest neighbor from second
        to first
    '''
    N = pc1.get_shape()[1].value
    M = pc2.get_shape()[1].value
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1, 2), [1, 1, M, 1])
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2, 1), [1, N, 1, 1])
    pc_diff = pc1_expand_tile - pc2_expand_tile  # B,N,M,C
    pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1)  # B,N,M
    dist1 = tf.reduce_min(pc_dist, axis=2)  # B,N
    idx1 = tf.argmin(pc_dist, axis=2)  # B,N
    dist2 = tf.reduce_min(pc_dist, axis=1)  # B,M
    idx2 = tf.argmin(pc_dist, axis=1)  # B,M
    return dist1, idx1, dist2, idx2
