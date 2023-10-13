import numpy as np
import matplotlib.pyplot as plt
import cv2
# Parameters, add more if needed
sigma  = 2    # Gaussian sigma 
thresh = 0.01 # harris thresh
size   = 4    # neighbour window size
k      = 0.04 # Harris detector free parameter in the equation.

# Derivative masks
dx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
dy = dx.transpose()

#? image transformation and plotting
img1 = cv2.cvtColor(cv2.imread('Harris-1.jpg'), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('Harris-2.jpg'), cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(cv2.imread('Harris-3.jpg'), cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(cv2.imread('Harris-4.jpg'), cv2.COLOR_BGR2GRAY)

img1_f      = np.float32(img1)
dst         = cv2.cornerHarris(img1_f, 2, 3, k)
CV2corners1 = []
for row in range(dst.shape[0]):
        for col in range(dst.shape[1]):
            if dst[row][col] >= thresh * dst.max():
                CV2corners1.append((row, col))
CV2corners1 = np.array(CV2corners1)

img2_f      = np.float32(img2)
dst         = cv2.cornerHarris(img2_f, 2, 3, k)
CV2corners2 = []
for row in range(dst.shape[0]):
        for col in range(dst.shape[1]):
            if dst[row][col] >= thresh * dst.max():
                CV2corners2.append((row, col))
CV2corners2 = np.array(CV2corners2)

img3_f      = np.float32(img3)
dst         = cv2.cornerHarris(img3_f, 2, 3, k)
CV2corners3 = []
for row in range(dst.shape[0]):
        for col in range(dst.shape[1]):
            if dst[row][col] >= thresh * dst.max():
                CV2corners3.append((row, col))
CV2corners3 = np.array(CV2corners3)

img4_f      = np.float32(img4)
dst         = cv2.cornerHarris(img4_f, 2, 3, k)
CV2corners4 = []
for row in range(dst.shape[0]):
        for col in range(dst.shape[1]):
            if dst[row][col] >= thresh * dst.max():
                CV2corners4.append((row, col))
CV2corners4 = np.array(CV2corners4)

fig_2       = plt.figure(figsize=(10, 10))
CV2img1plot = fig_2.add_subplot(2, 2, 1)
CV2img2plot = fig_2.add_subplot(2, 2, 2)
CV2img3plot = fig_2.add_subplot(2, 2, 3)
CV2img4plot = fig_2.add_subplot(2, 2, 4)

CV2img1plot.imshow(img1, cmap='gray')
CV2img1plot.scatter(CV2corners1[:,1], CV2corners1[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
CV2img2plot.imshow(img2, cmap='gray')
CV2img2plot.scatter(CV2corners2[:,1], CV2corners2[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
CV2img3plot.imshow(img3, cmap='gray')
CV2img3plot.scatter(CV2corners3[:,1], CV2corners3[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
CV2img4plot.imshow(img4, cmap='gray')
CV2img4plot.scatter(CV2corners4[:,1], CV2corners4[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')

CV2img1plot.set_xlabel('col')
CV2img1plot.set_ylabel('row')
CV2img1plot.set_title('image 1 (inbuilt Harris Corner dectector)')

CV2img2plot.set_xlabel('col')
CV2img2plot.set_ylabel('row')
CV2img2plot.set_title('image 2 (inbuilt Harris Corner dectector)')

CV2img3plot.set_xlabel('col')
CV2img3plot.set_ylabel('row')
CV2img3plot.set_title('image 3 (inbuilt Harris Corner dectector)')

CV2img4plot.set_xlabel('col')
CV2img4plot.set_ylabel('row')
CV2img4plot.set_title('image 4 (inbuilt Harris Corner dectector)')