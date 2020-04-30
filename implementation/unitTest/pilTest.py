from PIL import Image
from graphicUtils.image.utils import image_to_numpy
a = Image.open("/home/saurabh/Downloads/realImages/pistol/1.jpg")
a = a.resize((256, 192), Image.ANTIALIAS)
a.show()

# print(image_to_numpy(a, 255).shape)
import numpy as np
print(np.array(a).shape)
