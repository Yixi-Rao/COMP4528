# -*- coding: utf-8 -*-
# CLAB3 
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#
I = Image.open('stereo2012a.jpg');

#####################################################################
def Normalize(XYZ, h, w):
    lambdas, V = np.linalg.eig((XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)) @ (XYZ.T - np.mean(XYZ, axis=0).reshape(-1, 1)).T)
    S   = np.vstack((np.hstack((V @ np.diag(1 / lambdas) @ np.linalg.inv(V), -1 * V @ np.diag(1 / lambdas) @ np.linalg.inv(V) @ np.mean(XYZ, axis=0).reshape(-1, 1))),
                     np.hstack((np.zeros(3), np.ones(1)))
                    ))
    T_  = np.linalg.inv(np.array([[w + h, 0,     w / 2],
                                  [0,     w + h, h / 2],
                                  [0,     0,         1]]))
    return (S, T_)

def calibrate(im, XYZ, uv):  
    '''
        %% TASK 1: CALIBRATE
        %
        % Function to perform camera calibration
        %
        % Usage:   calibrate(image, XYZ, uv)
        %          return C
        %   Where:   image - is the image of the calibration target.
        %            XYZ - is a N x 3 array of  XYZ coordinates
        %                  of the calibration target points. 
        %            uv  - is a N x 2 array of the image coordinates
        %                  of the calibration target points.
        %            K   - is the 3 x 4 camera calibration matrix.
        %  The variable N should be an integer greater than or equal to 6.
        %
        %  This function plots the uv coordinates onto the image of the calibration
        %  target. 
        %
        %  It also projects the XYZ coordinates back into image coordinates using
        %  the calibration matrix and plots these points too as 
        %  a visual check on the accuracy of the calibration process.
        %
        %  Lines from the origin to the vanishing points in the X, Y and Z
        %  directions are overlaid on the image. 
        %
        %  The mean squared error between the positions of the uv coordinates 
        %  and the projected XYZ coordinates is also reported.
        %
        %  The function should also report the error in satisfying the 
        %  camera calibration matrix constraints.
        % 
        % Yixi Rao, 08/05/22 
    '''
    #*----------------------------------------------plotting----------------------------------------------------------------
    N     = XYZ.shape[0]
    S, T_ = Normalize(XYZ, im.shape[0], im.shape[1])

    h_XYZ = np.hstack((XYZ, np.ones(N).reshape(-1, 1)))
    h_uv  = np.hstack((uv,  np.ones(N).reshape(-1, 1)))
    n_XYZ = h_XYZ @ S.T #
    n_uv  = h_uv  @ T_.T #
    A     = None

    for i in range(N):
        Xi         = n_XYZ[i]
        xi, yi ,wi = n_uv[i]
        Ai         = np.vstack((np.hstack((np.zeros(4), -1 * wi * Xi,   yi * Xi     )), 
                                np.hstack((wi * Xi,     np.zeros(4),    -1 * xi * Xi))
                                ))
        A          = np.vstack((A, Ai)) if i != 0 else Ai
    
    _, _, v  = np.linalg.svd(A)
    C        = (v[-1] / np.sum(v[-1])).reshape(3, 4) #
    P_denorm = np.linalg.inv(T_) @ C @ S
    
    #*----------------------------------------------plotting----------------------------------------------------------------
    
    fig_1    = plt.figure(figsize=(20, 20))
    img1plot = fig_1.add_subplot(2, 1, 1)
    img1plot.imshow(im)
    img1plot.scatter(uv[:, 0], uv[:, 1],  s=40, marker="o", edgecolors="deeppink",facecolors='none')
    
    img1plot.set_xlabel('col')
    img1plot.set_ylabel('row')
    img1plot.set_title('original UV (stereo2012a.jpg)')
    
    projected_uv = h_XYZ @ P_denorm.T
    projected_uv = (projected_uv / projected_uv[:, -1].reshape(-1, 1))[:, 0 : 2]
    
    img1plot2 = fig_1.add_subplot(2, 1, 2)
    img1plot2.imshow(im)
    img1plot2.scatter(projected_uv[:, 0], projected_uv[:, 1],  s=40, marker="o", edgecolors="red",facecolors='none')
    
    img1plot2.set_xlabel('col')
    img1plot2.set_ylabel('row')
    img1plot2.set_title('projected UV (stereo2012a.jpg)')
    
    Error = np.sqrt(np.mean([np.sum((projected_uv[i] - uv[i]) ** 2) for i in range(N)]))
    return P_denorm, Error

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
############################################################################
def T_norm(h, w):
    return np.linalg.inv(np.array([[w + h, 0,     w / 2], [0,     w + h, h / 2], [0,     0,         1]]))

def MSE(oriUV, tranUV):
    return np.sqrt(np.mean([np.sum((oriUV[i] - tranUV[i]) ** 2) for i in range(oriUV.shape[0])]))

def homography(u2Trans, v2Trans, uBase, vBase):
    '''
        %% TASK 2: 
        % Computes the homography H applying the Direct Linear Transformation 
        % The transformation is such that 
        % p = np.matmul(H, p.T), i.e.,
        % (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
        % Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
        % deal the value of axis 
        %
        % INPUTS: 
        % u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
        % uBase, vBase - vectors with coordinates u and v of the original base image point p  
        % 
        % OUTPUT 
        % H - a 3x3 Homography matrix  
        % 
        % Yixi Rao, 11/05/2022 
    '''
    N      = uBase.shape[0]
    T1     = T_norm(370, 492)
    T2     = T_norm(370, 492)
    
    uvBase     = np.hstack((uBase, vBase))
    uv2Trans   = np.hstack((u2Trans, v2Trans))

    h_uvBase   = np.hstack((uvBase,   np.ones(N).reshape(-1, 1)))
    h_uv2Trans = np.hstack((uv2Trans, np.ones(N).reshape(-1, 1)))
    
    n_uvBase   = h_uvBase   @ T1.T #
    n_uv2Trans = h_uv2Trans @ T2.T #
    
    A = None
    for i in range(N):
        Xi         = n_uvBase[i]
        xi, yi ,wi = n_uv2Trans[i]
        Ai         = np.vstack((np.hstack((np.zeros(3), -1 * wi * Xi,   yi * Xi     )), 
                                np.hstack((wi * Xi,     np.zeros(3),    -1 * xi * Xi))
                                ))
        A          = np.vstack((A, Ai)) if i != 0 else Ai

    _, _, v  = np.linalg.svd(A)
    H = (v[-1] / np.sum(v[-1])).reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H 

############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

if __name__ == '__main__':
    # plt.imshow(I)
    # uv = plt.ginput(6) # Graphical user interface to get 6 points
    # np.save('uv', np.array(uv))
    part = "Q1" # Q2
    if part == "Q1":
        XYZ = np.load('XYZ.npy')
        uv  = np.load('uv.npy')
        
        img1     = cv2.cvtColor(cv2.imread('stereo2012a.jpg'), cv2.COLOR_BGR2RGB)
        P, error = calibrate(img1, XYZ, uv)
        K, R, t  = vgg_KR_from_P(P)
        _, _, v  = np.linalg.svd(P)
        camera   = (v[-1] / v[-1,-1])
        print(f"Matrix P:\n {P}\n")
        print(f"MSE: {error}\n")
        print(f"Matrix K:\n {np.array(K)}\nMatrix R:\n {np.array(R)}\nMatrix T:\n {np.array(t)}\n")
        
        print(f"fx : {K[0, 0]}, fy : {K[1, 1]}\n")
        print(f"camera center: {camera}\n")
        
        img1_half     = cv2.resize(img1, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)))
        uv_half       = np.load('uv_half.npy')
        P_half, e2    = calibrate(img1_half, XYZ,  uv_half)
        K_2, R_2, t_2 = vgg_KR_from_P(P_half)
        print(f"Matrix P:\n {P_half}\n")
        print(f"MSE: {e2}\n")
        print(f"Matrix K:\n {np.array(K_2)}\nMatrix R:\n {np.array(R_2)}\nMatrix T:\n {np.array(t_2)}\n")
        
        print(f"fx : {K_2[0, 0]}, fy : {K_2[1, 1]}\n")
        
    imgL = cv2.cvtColor(cv2.imread('Left.jpg'), cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(cv2.imread('Right.jpg'), cv2.COLOR_BGR2RGB)
    
    uvL = np.load('uvL.npy')
    uvR = np.load('uvR.npy')
    
    H   = homography(uvR[:, 0].reshape(-1,1), uvR[:, 1].reshape(-1,1), uvL[:, 0].reshape(-1,1), uvL[:, 1].reshape(-1,1))
    #!-----------------------------------------image left and image right-------------------------------------------------------------------------------------------

    fig_LR   = plt.figure(figsize=(20, 20))
    imgLplot = fig_LR.add_subplot(2, 1, 1)
    imgLplot.imshow(imgL)
    imgLplot.scatter(uvL[:, 0], uvL[:, 1],  s=60, marker="o", color="red")
    
    imgLplot.set_xlabel('col')
    imgLplot.set_ylabel('row')
    imgLplot.set_title('image Left.jpg')
    
    imgRplot = fig_LR.add_subplot(2, 1, 2)
    imgRplot.imshow(imgR)
    imgRplot.scatter(uvR[:, 0], uvR[:, 1],  s=60, marker="o", color="red")
    
    imgRplot.set_xlabel('col')
    imgRplot.set_ylabel('row')
    imgRplot.set_title('image Right.jpg')
    #!-----------------------------------------wraped image---------------------------------------------------------------------------------------------------------
    imgW     = cv2.warpPerspective(imgL, H, (imgL.shape[1], imgL.shape[0]))
    fig_w    = plt.figure(figsize=(10, 10))
    imgwplot = fig_w.add_subplot(1, 1, 1)
    imgwplot.imshow(imgW)
    
    imgwplot.set_xlabel('col')
    imgwplot.set_ylabel('row')
    imgwplot.set_title('warped Left image')
    
    uvLtran = np.hstack((uvL, np.ones(uvL.shape[0]).reshape(-1, 1)))
    uvLtran = uvLtran @ H.T
    uvLtran = (uvLtran / uvLtran[:, -1].reshape(-1, 1))[:, 0 : 2]
    print(f"MSE(original): {MSE(uvR, uvLtran)}")
    
    
    
    
    
