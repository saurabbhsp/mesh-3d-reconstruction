import numpy as np
from deformations.utility.mesh3d import mesh3d
from deformations.utility.bernstein import get_bernstein_polynomial


def get_min_max(x, *args, **kwargs):
        return np.min(x, *args, **kwargs), np.max(x, *args, **kwargs)


def stu_to_xyz(stu_points, stu_origin, stu_axes):
    return stu_origin + stu_points*stu_axes


def get_stu_control_points(dims):
    stu_lattice = mesh3d(*(np.linspace(0, 1, d+1) for d in dims),
                         dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points


def get_control_points(dims, stu_origin, stu_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, stu_origin, stu_axes)
    return xyz_points


def xyz_to_stu(xyz, origin, stu_axes):
    if stu_axes.shape == (3,):
        stu_axes = np.diag(stu_axes)
    assert(stu_axes.shape == (3, 3))
    s, t, u = stu_axes
    tu = np.cross(t, u)
    su = np.cross(s, u)
    st = np.cross(s, t)

    diff = xyz - origin
    stu = np.stack([
        np.dot(diff, tu) / np.dot(s, tu),
        np.dot(diff, su) / np.dot(t, su),
        np.dot(diff, st) / np.dot(u, st)
    ], axis=-1)
    return stu


def get_stu_params(xyz):
    minimum, maximum = get_min_max(xyz, axis=0)
    stu_origin = minimum
    stu_axes = maximum - minimum
    return stu_origin, stu_axes


def get_stu_deformation_matrix(stu, dims):
    """v is a matrix of shape(l+1, m+1, n+1, 3)
       with all possible i, j and k combinations and 3
       as 3 dimensions wrt to stu"""
    v = mesh3d(*(np.arange(0, d+1, dtype=np.int32) for d in dims),
               dtype=np.int32)
    v = np.reshape(v, (-1, 3))
    weights = get_bernstein_polynomial(n=np.array(dims, dtype=np.int32),
                                       v=v,
                                       x=np.expand_dims(stu, axis=-2))

    b = np.prod(weights, axis=-1)
    return b


def get_deformation_matrix(xyz, dims, stu_origin, stu_axis):
    stu = xyz_to_stu(xyz, stu_origin, stu_axis)
    return get_stu_deformation_matrix(stu, dims)


def get_ffd(xyz, dims, stu_origin=None, stu_axis=None):
    if stu_origin is None or stu_axis is None:
        print("Generating origin and axis")
        stu_origin, stu_axis = get_stu_params(xyz)
    b = get_deformation_matrix(xyz, dims, stu_origin, stu_axis)
    p = get_control_points(dims, stu_origin, stu_axis)
    return b, p
