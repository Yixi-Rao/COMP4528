# -*- coding: utf-8 -*-
# CLAB3 
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sympy import Matrix, Symbol
#
I = Image.open('stereo2012a.jpg');

#####################################################################
# def Normalize(XYZ, h, w):
#     lambdas, V = np.linalg.eig((XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)) @ (XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)).T)
#     S   = np.vstack((np.hstack((V @ np.diag(1 / lambdas) @ np.linalg.inv(V), -1 * V @ np.diag(1 / lambdas) @ np.linalg.inv(V) @ np.mean(XYZ, axis=0).reshape(-1, 1))),
#                      np.hstack((np.zeros(3), np.ones(1)))
#                     ))
#     T_  = np.linalg.inv(np.array([[w + h, 0,     w / 2],
#                                   [0,     w + h, h / 2],
#                                   [0,     0,         1]]))
#     return (S, T_)

def calibrate(im, XYZ, uv):
    N     = XYZ.shape[0]
    # S, T_ = Normalize(XYZ, im.shape[0], im.shape[1])

    h_XYZ = np.hstack((XYZ, np.ones(N).reshape(-1, 1)))
    h_uv  = np.hstack((uv,  np.ones(N).reshape(-1, 1)))
    # n_XYZ = h_XYZ @ S.T #
    # n_uv  = h_uv  @ T_.T #
    A     = None

    for i in range(N):
        Xi         = h_XYZ[i]
        xi, yi ,wi = h_uv[i]
        Ai         = np.vstack((np.hstack((np.zeros(4), -1 * wi * Xi.T, yi * Xi.T     )), 
                                np.hstack((wi * Xi.T,   np.zeros(4),    -1 * xi * Xi.T))
                                ))
        A          = np.vstack((A, Ai)) if i != 0 else Ai
        
    _, _, v = np.linalg.svd(A)
    C       = (v[-1] / np.sum(v[-1])).reshape(3, 4) #
    
    #*----------------------------------------------------------------------------------------------------------------
    
    fig_1    = plt.figure(figsize=(20, 20))
    img1plot = fig_1.add_subplot(2, 1, 1)
    img1plot.imshow(im)
    img1plot.scatter(uv[:, 0], uv[:, 1],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    
    img1plot.set_xlabel('col')
    img1plot.set_ylabel('row')
    img1plot.set_title('original UV (stereo2012a.jpg)')
    
    # P_denorm     = np.linalg.inv(T_) @ C @ S
    projected_uv = h_XYZ @ C.T
    projected_uv = (projected_uv / projected_uv[:, -1].reshape(-1, 1))[:, 0 : 2]

    img1plot2 = fig_1.add_subplot(2, 1, 2)
    img1plot2.imshow(im)
    img1plot2.scatter(projected_uv[:, 0], projected_uv[:, 1],  s=40, marker="o", edgecolors="red",facecolors='none')
    
    img1plot2.set_xlabel('col')
    img1plot2.set_ylabel('row')
    img1plot2.set_title('projected UV (stereo2012a.jpg)')
    
    Error = np.sqrt(np.mean([np.sum((projected_uv[i] - uv[i]) ** 2) for i in range(N)]))
    return C, Error

def vgg_rq(S):
    S = S.T
    [Q,U] = np.linalg.qr(S[::-1,::-1], mode='complete')
    
    Q = Q.T
    Q = Q[::-1, ::-1]
    U = U.T
    U = U[::-1, ::-1]
    
    if np.linalg.det(Q)<0:
        U[:,0] = -U[:,0]
        Q[0,:] = -Q[0,:]
    return U,Q

def vgg_KR_from_P(P, noscale = True):
    N = P.shape[0]
    H = P[:,0:N]
    # print(N,'|', H)
    [K,R] = vgg_rq(H)
    if noscale:
        K = K / K[N-1,N-1]
        if K[0,0] < 0:
            D = np.diag([-1, -1, 1]);
            K = K @ D
            R = D @ R
        
            test = K*R; 
            assert (test/test[0,0] - H/H[0,0]).all() <= 1e-07
    
    t = np.linalg.inv(-P[:,0:N]) @ P[:,-1]
    return K, R, t

if __name__ == '__main__':
    # plt.imshow(I)
    # uv = plt.ginput(6) # Graphical user interface to get 6 points
    # np.save('uv', np.array(uv))

    XYZ = np.load('XYZ.npy')
    uv  = np.load('uv.npy')
    
    img1     = cv2.cvtColor(cv2.imread('stereo2012a.jpg'), cv2.COLOR_BGR2RGB)
    P, error = calibrate(img1, XYZ, uv)
    K, R, t  = vgg_KR_from_P(P)
    print(f"Matrix P:\n {P}\n")
    print(f"MSE: {error}\n")
    print(f"Matrix K:\n {np.array(K)}\nMatrix R:\n {np.array(R)}\nMatrix T:\n {np.array(t)}\n")
    
    print(f"fx : {K[0, 0]}, fy : {K[1, 1]}\n")
    
    