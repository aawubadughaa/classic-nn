import numpy as np
from PIL import Image

img = Image.open('capture1.png').convert('RGBA')
arr = np.array(img)
arr.resize(28,28)
print(arr)