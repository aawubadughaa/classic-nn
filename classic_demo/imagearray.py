import numpy as np
from PIL import Image
import torch
import pdb
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('capture1.png')

plt.figure(1)
plt.imshow(img)

res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
imgtor = torch.from_numpy(res)

plt.figure(2)
plt.imshow(res)

plt.show()