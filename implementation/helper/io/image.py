from PIL import Image


def load_image_from_file(f):
    return Image.open(f)


def load_image_from_zip(zip_file, path):
    with zip_file.open(path) as fp:
        return load_image_from_file(fp)

def load_from_file(f):
    with f:
        return f.read()

def load_resized_image_from_file(f, resolution):
    return Image.open(f).resize((resolution[1], resolution[0]), Image.ANTIALIAS)
