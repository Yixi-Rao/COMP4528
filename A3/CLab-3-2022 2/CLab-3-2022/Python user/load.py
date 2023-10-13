import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
XYZ = np.array([[0, 28, 14],
                [0, 21, 21],
                [14, 0, 14],
                [21, 0, 14],
                [7,  7, 0],
                [21, 42, 0]])
# np.save('XYZ', XYZ)
# I = Image.open('stereo2012a.jpg');
# plt.imshow(I)
# uv = plt.ginput(6) # Graphical user interface to get 6 points
# np.save('uv', uv)
# print(np.load('uv.npy'))
# print (np.load('XYZ.npy'))

# img1      = cv2.cvtColor(cv2.imread('stereo2012a.jpg'), cv2.COLOR_BGR2RGB)
# img1_half = cv2.resize(img1, (int(img1.shape[1] / 2 ), int(img1.shape[0] / 2)))
# plt.imshow(img1_half)
# uv_half = plt.ginput(6) # Graphical user interface to get 6 points
# np.save('uv_half', uv_half)
# print(np.load('uv_half.npy'))

#!------------------------------------------------------

# I = Image.open('Left.jpg');
# plt.imshow(I)
# uvL = plt.ginput(6, timeout=300)
# np.save('uvL', uvL)

# I = Image.open('Right.jpg');
# plt.imshow(I)
# uvR = plt.ginput(6, timeout=300)
# np.save('uvR', uvR)

# print(np.load('uvL.npy'))
# print(np.load('uvR.npy'))

# imgL = cv2.cvtColor(cv2.imread('Left.jpg'), cv2.COLOR_BGR2RGB)
# imgR = cv2.cvtColor(cv2.imread('Right.jpg'), cv2.COLOR_BGR2RGB)

I = Image.open('Left.jpg');
plt.imshow(I)
uvtL = plt.ginput(8, timeout=300)
np.save('uvtL', uvtL)

# I2 = Image.open('Right.jpg');
# plt.imshow(I2)
# uvtR = plt.ginput(8, timeout=300)
# np.save('uvtR', uvtR)

