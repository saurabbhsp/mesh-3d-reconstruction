
"""
Utility methods for processing voxel processing
"""


def read_header(fp):
    """
    The .binvox file has following format.
    This is metadata for the binvoxel file.

    binvox
    #binvox 1
    dim 32 32 32
    translate -0.5 -0.5 -0.5
    scale 1

    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = tuple(int(s) for s in fp.readline().strip().split(b' ')[1:])
    translate = tuple(float(s) for s in fp.readline().strip().split(b' ')[1:])
    scale = float(fp.readline().strip().split(b' ')[1])
    fp.readline()
    """The above file pointer read data string in binvox file. Now
    pointing to the datapoint"""
    return dims, translate, scale
