#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def rot_3D(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def PCA(points):

    # Compute the barycenter
    center = np.mean(points, axis=0)
    
    # Centered clouds
    points_c = points - center

    # Covariance matrix
    C = (points_c.T).dot(points_c) / points.shape[0]

    # Eigenvalues
    return np.linalg.eigh(C)


def compute_local_PCA(cloud, radius):

    # Create a neighbor structure on the cloud
    print('Constructing cloud')
    t0 = time.time()
    search_tree = KDTree(cloud)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Find the nearest neighbors
    print('Find neighbors')
    t0 = time.time()
    indices = search_tree.query_radius(cloud, radius)
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Initiate normals
    all_eigenvalues = np.empty(cloud.shape, dtype=np.float32)
    all_eigenvectors = np.empty((cloud.shape[0], 3, 3), dtype=np.float32)

    print('Compute all PCA')
    t0 = time.time()
    for i, neighbors_inds in enumerate(indices):
        all_eigenvalues[i, :], all_eigenvectors[i, :, :] = PCA(cloud[neighbors_inds, :])
    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    return all_eigenvalues, all_eigenvectors


def compute_features(cloud, radius):

    # Compute PCA on all cloud
    all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, radius)
    normals = all_eigenvectors[:, :, 0]

    # Compute verticality
    ez = np.array([0, 0, 1]).reshape(-1, 1)
    verticality = np.arcsin(np.abs(normals.dot(ez))) * 180 / np.pi

    # Compute features
    linearity = 1 - all_eigenvalues[:,1] / (all_eigenvalues[:,2]+1e-9)
    planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,0]) / (all_eigenvalues[:,2]+1e-9)
    sphericity = all_eigenvalues[:,0] / (all_eigenvalues[:,2]+1e-9)

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, 0.5)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute features on the whole cloud
        verticality, linearity, planarity, sphericity = compute_features(cloud, 0.5)

        # Save cloud with normals
        write_ply('../Lille_street_small_features.ply', (cloud, verticality, linearity, planarity, sphericity), ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
