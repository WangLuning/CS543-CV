from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt

##
## load images and match files for the first example
##
dirs = 'MP3_part2_data/'
I1 = Image.open(dirs + 'library1.jpg');
I2 = Image.open(dirs + 'library2.jpg');
matches = np.loadtxt(dirs + 'library_matches.txt'); 

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image: 
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
I3[:,:I1.size[0],:] = I1;
I3[:,I1.size[0]:,:] = I2;
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I3).astype(np.uint8))
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
plt.show()

##
## display second image with epipolar lines reprojected 
## from the first image
##

def fit_fundamental(matches, normalize = True):     
    # Solve homogeneous linear system using eight or more matches  
    # no need to change to homogeneous style since we reform it to svd form later 
    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]

    # normalize the points if we use normalized eight points algo
    if normalize:
        p1, T1 = normalization(p1)
        p2, T2 = normalization(p2)

    # select randomly eight points to perform the algo
    rand_idx = random.sample(range(p1.shape[0]), k=8)
    eight_p1 = p1[rand_idx]
    eight_p2 = p2[rand_idx]

    # fitting the fundamental using eight point algo to solve F matrix
    A = []
    for i in range(eight_p1.shape[0]):
        p1 = eight_p1[i]
        p2 = eight_p2[i]
        
        row = [p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1]
        A.append(row)

    A = np.array(A)

    U, s, V = np.linalg.svd(A)
    F = V[len(V)-1].reshape(3, 3)
    # normalized F
    F = F / F[2, 2] 

    # Enforce rank-2 constraint.
    U, s, v = np.linalg.svd(F)
    # Vector(s) with the singular values, within each vector sorted in descending order
    s_throw_out = np.diag(s)
    # throw out the smallest value
    s_throw_out[-1] = 0
    F = np.dot(U, np.dot(s_throw_out, v))

    # recover the unnormalized F matrix
    if normalize:
        F = np.dot(np.dot(T2.T, F), T1)
    
    residual = calculate_residual(matches, F)
    print('residual of method: ' + str(residual))
    return F

def normalization(pts):
    """Helper function to normalized data in image."""
    # "Center the image data at the origin". 
    # You can do this by just subtracting the mean of the data from each point.
    mean = np.mean(pts, axis=0)
    pts_x_centered = pts[:, 0] - mean[0]
    pts_y_centered = pts[:, 1] - mean[1]

    #Scale so the mean squared distance between origin and data point is 2
    scale = sqrt(1 / (2 * len(pts)) * np.sum(pts_x_centered**2 + pts_y_centered**2))
    scale = 1 / scale

    transform = np.array([[scale, 0, -scale*mean[0]], 
                           [0, scale, -scale*mean[1]], 
                           [0, 0, 1]])
    # do homogeneous transform
    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    normalized = np.dot(transform, pts.T).T

    return normalized[:, 0:2], transform

def calculate_residual(matches, F):
    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]
    p1_homo = np.concatenate((p1, np.ones((p1.shape[0], 1))), axis=1)
    p2_homo = np.concatenate((p2, np.ones((p2.shape[0], 1))), axis=1)

    residual = 0
    for i in range(p1.shape[0]):
        residual += abs(np.dot(np.dot(p2_homo[i], F), p1_homo[i].T))

    residual = residual / matches.shape[0]
    return residual

# first, fit fundamental matrix to the matches
F = fit_fundamental(matches); # this is a function that you should write
M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
L1 = np.matmul(F, M).transpose() # transform points from 
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(np.uint8))
ax.plot(matches[:,2],matches[:,3],  '+r')
ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
plt.show()


