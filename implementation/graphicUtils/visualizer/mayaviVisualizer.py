import numpy as np
from mayavi import mlab


"""Generate the permutations
for the coordinates subject to the
order"""


def permute_xyz(x, y, z, order='xyz'):
    _dim = {'x': 0, 'y': 1, 'z': 2}
    data = (x, y, z)
    return tuple(data[_dim[k]] for k in order)


@mlab.animate
def visualize_point_cloud(points, axis_order='xyz', value=None, **kwargs):
    data = permute_xyz(*points.T, order=axis_order)
    if value is not None:
        data = data + (value,)
    scene1 = mlab.points3d(*data, **kwargs)
    while 1:
        scene1.scene.camera.azimuth(10)
        scene1.scene.render()
        yield points, axis_order, value, kwargs


@mlab.animate
def visualize_voxels(voxels, axis_order='xzy', **kwargs):
    data = permute_xyz(*np.where(voxels), order=axis_order)
    if len(data[0]) == 0:
        print('No voxels to display')
    else:
        if 'mode' not in kwargs:
            kwargs['mode'] = 'cube'
        scene1 = mlab.points3d(*data, **kwargs)
        while 1:
            scene1.scene.camera.azimuth(10)
            scene1.scene.render()
            yield voxels, axis_order, kwargs


@mlab.animate
def visualize_mesh(
        vertices, faces, axis_order='xyz', include_wireframe=True,
        color=(0, 0, 1), **kwargs):
    if len(faces) == 0:
        print('Warning: no faces')
        return
    x, y, z = permute_xyz(*vertices.T, order=axis_order)
    scene1 = mlab.triangular_mesh(x, y, z, faces, color=color, **kwargs)
    if include_wireframe:
        scene2 = mlab.triangular_mesh(
            x, y, z, faces, color=(0, 0, 1), representation='wireframe')

    while 1:
        scene1.scene.camera.azimuth(10)
        if include_wireframe:
            scene2.scene.camera.azimuth(10)
        scene1.scene.render()
        if include_wireframe:
            scene2.scene.render()
        yield vertices, faces, axis_order, include_wireframe, color, kwargs


@mlab.animate
def visualize_normals(positions, normals, axis_order='xyz', **kwargs):
    x, y, z = permute_xyz(*positions.T, order=axis_order)
    u, v, w = permute_xyz(*normals.T, order=axis_order)
    scene1 = mlab.quiver3d(x, y, z, u, v, w, **kwargs)
    while 1:
        scene1.scene.camera.azimuth(10)
        scene1.scene.render()
        yield positions, normals, axis_order, kwargs
