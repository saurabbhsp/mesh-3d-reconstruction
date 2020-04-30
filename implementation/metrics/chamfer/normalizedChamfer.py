import numpy as np
from scipy.optimize import minimize
from metrics.chamfer.chamfer_numpy import NumpyMetrics

def normalized(points, offset, scale_factor):
    return (points - offset) / scale_factor

def get_normalization_params(vertices):
    vertices = np.array(vertices)
    vertical_offset = np.min(vertices[:, 1])
    vertices[:, 1] -= vertical_offset

    def f(x):
        x = np.array([x[0], 0, x[1]])
        dist2 = np.sum((vertices - x)**2, axis=-1)
        return np.max(dist2)

    opt = minimize(f, np.array([0, 0])).x
    offset = np.array([opt[0], vertical_offset, opt[1]], dtype=np.float32)
    vertices[:, [0, 2]] -= opt

    radius = np.sqrt(np.max(np.sum(vertices**2, axis=-1)))
    unit1 = 3.2
    scale_factor = radius / unit1
    return offset, scale_factor


def get_normalized_chamfer(ground_truth_mesh, ground_truth_point_cloud,
                           deformation, n_samples):
    normalized_chamfer = []
    default_chamfer = []
    for _ground_truth_mesh, _ground_truth_point_cloud, _deformation in zip(ground_truth_mesh,
                                                ground_truth_point_cloud,
                                                deformation):
        offset, scale_factor = get_normalization_params(_ground_truth_mesh['vertices'])

        _normalized_gt = normalized(_ground_truth_point_cloud, offset, scale_factor)
        _normalized_deformation = normalized(_deformation, offset, scale_factor)
        normalized_chamfer.append(NumpyMetrics().chamfer(_normalized_gt, _normalized_deformation)/n_samples)
        default_chamfer.append(NumpyMetrics().chamfer(_deformation, _ground_truth_point_cloud)/n_samples)
    return normalized_chamfer, default_chamfer
