import numpy as np


def read_raw_mesh(open_file):
    positions = []
    texcoords = []
    normals = []
    face_positions = []
    face_texcoords = []
    face_normals = []

    def parse_face(values):
        if len(values) != 3:
            raise ValueError('not a triangle at line' % lineno)
        for v in values:
            for j, index in enumerate(v.split('/')):
                if len(index):
                    if j == 0:
                        face_positions.append(int(index) - 1)
                    elif j == 1:
                        face_texcoords.append(int(index) - 1)
                    elif j == 2:
                        face_normals.append(int(index) - 1)

    parse_fns = {
        'v': lambda values: positions.append([float(x) for x in values]),
        'vt': lambda values: texcoords.append([float(x) for x in values]),
        'vn': lambda values: normals.append([float(x) for x in values]),
        'f': parse_face,
        'mtllib': lambda values: None,
        'o': lambda values: None,
        'usemtl': lambda values: None,
        's': lambda values: None,
        'newmtl': lambda values: None,
        'Ns': lambda values: None,
        'Ni': lambda values: None,
        'Ka': lambda values: None,
        'Kd': lambda values: None,
        'Ks': lambda values: None,
        'd': lambda values: None,
        'illum': lambda values: None,
        'map_Kd': lambda values: None,
    }

    def parse_line(line):
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            values = line.split(' ')
            code = values[0]
            values = values[1:]
            if code in parse_fns:
                parse_fns[code](values)

    for lineno, line in enumerate(open_file.readlines()):
        parse_line(line.decode("utf-8"))

    """Processing read data"""
    positions = np.array(positions, dtype=np.float32)
    if len(texcoords) > 0:
        texcoords = np.array(texcoords, dtype=np.float32)
    else:
        texcoords = None

    if len(normals) > 0:
        normals = np.array(normals, dtype=np.float32)
    else:
        normals = None

    face_positions = np.array(face_positions, dtype=np.uint32).reshape(-1, 3)

    if len(face_texcoords) > 0:
        face_texcoords = np.array(face_texcoords,
                                  dtype=np.uint32).reshape(-1, 3)
    else:
        face_texcoords = None

    if len(face_normals) > 0:
        face_normals = np.array(face_normals,
                                dtype=np.uint32).reshape(-1, 3)
    else:
        face_normals = None
    print("Processed")
    return positions, face_positions, texcoords, face_texcoords, \
        normals, face_normals


def sample_triangle(v, n=None):
    if hasattr(n, 'dtype'):
        n = np.asscalar(n)
    if n is None:
        size = v.shape[:-2] + (2,)
    elif isinstance(n, int):
        size = (n, 2)
    elif isinstance(n, tuple):
        size = n + (2,)
    elif isinstance(n, list):
        size = tuple(n) + (2,)
    else:
        raise TypeError('n must be int, tuple or list, got %s' % str(n))
    assert(v.shape[-2] == 2)
    a = np.random.uniform(size=size)
    mask = np.sum(a, axis=-1) > 1
    a[mask] *= -1
    a[mask] += 1
    a = np.expand_dims(a, axis=-1)
    return np.sum(a*v, axis=-2)


def sample_mesh_faces(vertices, faces, n_total):

    """
    Each pair of triangle_coordinates has three points
    Each point is three dimensional value representing coordinate
    in x, y and z coordinates
    Essentially each face will be
    [[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]]
    """
    triangle_coordinates = vertices[faces]
    n_faces = len(faces)

    """
    Step 1 - Select first point ie d0
    Step 2 - Break geometry
    """

    d0 = triangle_coordinates[..., 0:1, :]
    ds = triangle_coordinates[..., 1:, :] - d0

    """Need to check this part"""
    assert(ds.shape[1:] == (2, 3))
    areas = 0.5 * np.sqrt(np.sum(np.cross(ds[:, 0], ds[:, 1])**2, axis=-1))
    cum_area = np.cumsum(areas)
    cum_area *= (n_total / cum_area[-1])
    cum_area = np.round(cum_area).astype(np.int32)

    positions = []
    last = 0
    for i in range(n_faces):
        n = cum_area[i] - last
        last = cum_area[i]
        if n > 0:
            positions.append(d0[i] + sample_triangle(ds[i], n))
    return np.concatenate(positions, axis=0)


def sample_mesh_faces_normals(vertices, faces, n_total):
        if len(faces) == 0:
            raise ValueError('Cannot sample points from zero faces.')
        tris = vertices[faces]
        d0 = tris[..., 0:1, :]
        ds = tris[..., 1:, :] - d0
        d0 = np.squeeze(d0, axis=-2)
        assert(ds.shape[1:] == (2, 3))
        normals = np.cross(ds[:, 0], ds[:, 1])
        norm = np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
        areas = np.squeeze(norm, axis=-1).copy()
        total_area = np.sum(areas)
        areas /= total_area * (1 + 1e-3)
        norm_eps = 1e-8
        norm[norm < norm_eps] = norm_eps
        normals /= norm

        counts = np.random.multinomial(n_total, areas)
        indices = np.concatenate(
            tuple((i,)*c for i, c in enumerate(counts)),
            axis=0).astype(np.int32)
        positions = d0[indices] + sample_triangle(ds[indices])
        normals = normals[indices]
        return positions, normals
