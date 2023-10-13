import cv2
import numpy as np

# XYZ = np.load('XYZ.npy')
XYZ = np.array([[0,0,0],[7,7,0],[0,14,7],[14,0,14],[14,14,0],[0,14,14]])
uv  = np.load('uv.npy')

img1    = cv2.imread('stereo2012a.jpg')
h, w, _ = img1.shape

def Normalize(XYZ, h, w):
    lambdas, V = np.linalg.eig((XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)) @ (XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)).T)
    S = np.vstack((np.hstack((V @ np.diag(1 / lambdas) @ np.linalg.inv(V), -1 * V @ np.diag(1 / lambdas) @ np.linalg.inv(V) @ np.mean(XYZ, axis=0).reshape(-1, 1))),
                   np.hstack((np.zeros(3),                                 np.ones(1)))
                  ))
    
    T  = np.linalg.inv(np.array([[w + h, 0,     w / 2],
                                 [0,     w + h, h / 2],
                                 [0,     0,         1]]))
    return (S, np.linalg.inv(T))

# print(np.hstack((XYZ, 2 * np.ones(6).reshape(-1, 1))))
# a = np.hstack((XYZ, 2 * np.ones(6).reshape(-1, 1)))
# print((a / a[:, -1].reshape(-1, 1))[:, 0 : 3])
# print(np.mean(XYZ, axis=0))
# print("---------------------------")

S, T = Normalize(XYZ, h, w)
# print(np.hstack((XYZ, np.ones(6).reshape(-1, 1))).shape)
# print(S.T.shape)
# np.hstack((XYZ, np.ones(6).reshape(-1, 1))) @ S.T