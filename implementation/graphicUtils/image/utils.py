import numpy as np

"""Here the input image is
RGBalpha (png image) The following method will
convert RGBalpha to RGB image. This process can be
skipped for JPEG images"""


def image_to_numpy(image, background_color):
    """Sets background of 4d image to the specified color."""
    image = np.asarray(image)
    assert(image.shape[-1] == 4)
    background = image[..., 3] == 0
    image = image[..., :3].copy()
    image[background] = background_color
    return image
