# ------------------------------------------------------------------------------------------
#
#      Jiadong WANG - 25/06/2019

# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

# Import sys package
import sys

# Import scikit-image
from skimage import measure

##  RANSAC #########################################
# compute normal vector of a plane with three points
def compute_plane(points):
    
    ref_point = points[0]
    normal = np.cross(points[1]-points[0], points[2]-points[0])
    normal = normal / np.sqrt(np.sum(np.power(normal,2)))
    return ( ref_point, normal )

# judge if points on a specific plane with a threshold
def in_plane(points, ref_pt, normal, threshold_in=0.2):

    return (np.abs(np.dot((points - ref_pt) , normal)) < threshold_in)

# RANSAC to find the biggest plane
def RANSAC(points, NB_RANDOM_DRAWS=150, threshold_in=0.1):
    
    best_vote = 3
    best_ref_pt, best_normal = compute_plane(points[:3])
    N = len(points)
    
    for i in range(NB_RANDOM_DRAWS):
        
        random_inds = np.random.randint(0, N, size=3)
        ref_pt, normal = compute_plane(points[random_inds])
        
        vote = np.sum(in_plane(points, ref_pt, normal, threshold_in))
 
        if vote > best_vote:
            best_ref_pt = ref_pt
            best_normal = normal
            best_vote = vote
    print("best_vote: " + str(best_vote))
            
    return best_ref_pt, best_normal, best_vote


if __name__ == '__main__':

    # Load .ply file and process as numpy ndarray
    pass

    # Extraction of ground by RANSAC
    print('RANSAC begin: ')
    best_ref_pt, best_normal, best_vote = RANSAC(points)
    points_out_of_plane = points[~in_plane(points[i], best_ref_pt, best_normal, threshold_in=0.2)]
    print('RANSAC end. ')

    # Find the upward normal of plane
    if np.sum(np.dot(points_out_of_plane[:]-best_ref_pt,best_normal.T)>0)>points_out_of_plane.shape[0]/2:
        pass
    else:
        best_normal = best_normal*(-1)

    # Compute the rotation matrix and rotate the point cloud
    # [x' y' z'].T=R X[x y z].T
    # [0 0 1]=R Xbest_normal.T
    thetax=0 
    thetay=0 
    thetaz=0 
    R = np.array([[],
                  [],
                  []])
    points = np.dot(R,points.T)

    # Compute the translation vector (put the center of ground at (0,0,0)) and translate the point cloud
    d = -0.5*(np.max(points, axis=0)+np.min(points, axis=0))
    points = points+d
    
    #