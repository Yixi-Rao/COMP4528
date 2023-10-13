"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Yixi Rao (u6826541)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def conv2(img : np.array, conv_filter : np.array) -> np.array:
    '''convolution funciton

        Args:
            img (np.array): image
            conv_filter (np.array): filter 

        Returns:
            np.array: result
    '''
    # flip the filter
    f_siz_1, _  = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad         = (conv_filter.shape[0] - 1) // 2
    result      = np.zeros((img.shape))
    img         = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region  = img[r : r + filter_size, c : c + filter_size]
            curr_result  = curr_region * conv_filter
            conv_sum     = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum             # Saving the summation in the convolution layer feature map.

    return result

def fspecial(shape=(3, 3), sigma=0.5):
    '''compute a normalised Gaussain kernal, where the kernal size is related to the sigma, namely, kernal_size = 2 * floor(3 * sigma) + 1

        Args:
            shape (tuple, optional): filter size. Defaults to (3, 3).
            sigma (float, optional): Guassian sigma. Defaults to 0.5.

        Returns:
            _type_: Gaussian filter
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h    = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def rgb2gray(rgb : np.array) -> np.array:
    '''if using the plt.imread then we have to calculate the grey style image manually using the equation: grey = 0.299*R + 0.587*G + 0.114*B

        Args:
            rgb (np.array): rgb image

        Returns:
            np.array: grey style image
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def compute_M(bw : np.array, dx : np.array, dy : np.array) -> tuple:
    ''' compute the M matrix, where M is a 2x2 matrix computed from image derivatives

        Args:
            bw (np.array): image
            dx (np.array): kenel dx
            dy (np.array): kenel dy

        Returns:
            tuple: (Iy2, Ix2, Ixy)
    '''
    # computer x and y derivatives of image
    Ix = conv2(bw, dx) 
    Iy = conv2(bw, dy)
    #! BLOCK #5
    # compute a normalised Gaussian kernel g using the fspecial function, where the kernel size is related to the sigma, namely, kernal_size = 2 * floor(3 * sigma) + 1
    g   = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
    #! BLOCK #5 end
    Iy2 = conv2(np.power(Iy, 2), g)
    Ix2 = conv2(np.power(Ix, 2), g)
    Ixy = conv2(Ix * Iy, g)
    return (Iy2, Ix2, Ixy)

#! BLOCK #7
######################################################################
# Task: Compute the Harris Cornerness
######################################################################
def Harris_Cornerness(Ix2 : np.array, Iy2 : np.array, Ixy : np.array, k : float = 0.04) -> np.array:
    '''Compute the Harris Cornerness, using the fact:
       1. det(M)   = Ix2 * Iy2 - Ixy ** 2
       2. Trace(M) = (Ix2 + Iy2)
       3. R        = det(m) - (k * (trace(m)) ^ 2)
        Args:
            Ix2 (np.array): Ix**2
            Iy2 (np.array): Iy**2
            Ixy (np.array): Ix * Iy
            k (float)     : Harris detector free parameter in the equation.

        Returns:
            np.array: R matrix
    '''
    return (Ix2 * Iy2 - Ixy ** 2) - k * ((Ix2 + Iy2) ** 2)

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
def find_local_max(x : int, y : int, matirx : np.array, size : int = 1) -> tuple:
    '''using a neighbour window size to find all the adjacent points of (x,y), and find the maximum R value among it

        Args:
            x (int): centre point x
            y (int): centre point y
            matirx (np.array): what we perform on
            size (int, optional): neighbour window size. Defaults to 1.

        Returns:
            tuple: this tuple contains three values, this first is the R value, second and third are the coordinates of the neighbour point of this R value
    '''
    xmax, ymax = matirx.shape # boundary of x and y
    neighbours = [] # list of tuples: (R value, X, Y)
    # going through all the neighbour's R values
    for xi in range(-1 * size, size + 1):
        if x + xi >= 0 and x + xi <= xmax - 1: # all neighbour points should satisfy 0 <= x + xi <= xmax - 1
            for yi in range(-1 * size, size + 1):
                if y + yi >= 0 and y + yi <= ymax - 1: # all neighbour points should satisfy 0 <= y + yi <= ymax - 1
                    neighbours.append((matirx[x + xi][y + yi], x + xi, y + yi))

    return max(neighbours) # finding the maximum neighbour point
    
def detect_Harris_Corner(R : np.array, thresh : float = 0.01, neighbour_size : int = 1) -> np.array:
    '''detect all the Corners by using a thresh and non-maximum suppression

        Args:
            R (np.array): R value matrix
            thresh (float, optional): R thresh. Defaults to 0.01.
            neighbour_size (int, optional): used to decide the neighbour point's range in NMS. Defaults to 1.

        Returns:
            np.array: an Nx2 matrix of all the Corners
    '''
    rows, cols    = R.shape
    corner_points = []  # contains all the Corners' coordinates     
    R_max         = np.max(R) # maximum Response in R matrix
    for row in range(rows):
        for col in range(cols):
            if R[row][col] >= thresh * R_max: # if this point is a corner then R must be greater than thresh * R_max 
                _, xmax, ymax = find_local_max(row, col, R, neighbour_size) # all the adjacent points of (row, col)
                if (xmax, ymax) == (row, col): # if (row, col) is the local maxima of its all adjacent points then add this to the corners list
                    corner_points.append((row, col))
    return np.array(corner_points)            

def Harris_Corner(img : np.array, dx : np.array, dy : np.array, k : float = 0.04) -> np.array:
    '''overall function, integrate the compute_M, Harris_Cornerness, detect_Harris_Corner functions into one function,
       to calculate tbe corners

        Args:
            img (np.array): orignal image
            dx (np.array): Derivative masks
            dy (np.array): Derivative masks

        Returns:
            np.array: an Nx2 matrix of all the Corners
    '''
    # calculate the Iy^2, Ix^2, Ix*Iy
    Iy2, Ix2, Ixy = compute_M(img, dx, dy)
    # calculate the R matrix and collect all the corners
    R       = Harris_Cornerness(Ix2, Iy2, Ixy, k)
    corners = detect_Harris_Corner(R, thresh, size)
    return corners
#! BLOCK #7 end

if __name__ == '__main__':
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

    corners1 = Harris_Corner(img1, dx, dy, k)
    corners2 = Harris_Corner(img2, dx, dy, k)
    corners3 = Harris_Corner(img3, dx, dy, k)
    corners4 = Harris_Corner(img4, dx, dy, k)
    
    fig_1    = plt.figure(figsize=(10, 10))
    img1plot = fig_1.add_subplot(2, 2, 1)
    img2plot = fig_1.add_subplot(2, 2, 2)
    img3plot = fig_1.add_subplot(2, 2, 3)
    img4plot = fig_1.add_subplot(2, 2, 4)
    
    img1plot.imshow(img1, cmap='gray')
    img1plot.scatter(corners1[:,1], corners1[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    img2plot.imshow(img2, cmap='gray')
    img2plot.scatter(corners2[:,1], corners2[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    img3plot.imshow(img3, cmap='gray')
    img3plot.scatter(corners3[:,1], corners3[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    img4plot.imshow(img4, cmap='gray')
    img4plot.scatter(corners4[:,1], corners4[:,0],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    
    img1plot.set_xlabel('col')
    img1plot.set_ylabel('row')
    img1plot.set_title('image 1 (my Harris Corner dectector)')
    
    img2plot.set_xlabel('col')
    img2plot.set_ylabel('row')
    img2plot.set_title('image 2 (my Harris Corner dectector)')
    
    img3plot.set_xlabel('col')
    img3plot.set_ylabel('row')
    img3plot.set_title('image 3 (my Harris Corner dectector)')
    
    img4plot.set_xlabel('col')
    img4plot.set_ylabel('row')
    img4plot.set_title('image 4 (my Harris Corner dectector)')
    
    #? inbuilt Harris corner detector
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