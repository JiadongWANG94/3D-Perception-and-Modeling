print("OK")
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

from tqdm import tqdm
# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def PCA(points):
    
    G = np.mean(points,axis=0)
    Mcov = np.zeros([3,3])
    
    for i in range(np.shape(points)[0]):
        #print(np.shape(points[i]-G))
        Mcov = Mcov + np.dot(np.reshape((points[i]-G),[3,1]),np.reshape(points[i]-G,[1,3]))
#        if i==1: 
#            print((points[i]-G).T) 
#            print(points[i]-G) 
    Mcov=Mcov/np.shape(points)[0]
    eigenvalues, eigenvectors = np.linalg.eigh(Mcov)
    #print(eigenvalues)

    return eigenvalues, eigenvectors


def compute_local_PCA(cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))
    tree=KDTree(cloud_points)
    for i in tqdm(range(cloud_points.shape[0])):
        neighbor_index = tree.query_radius(np.reshape(cloud_points[i],[1,3]), radius)
        neighbor = np.zeros([np.shape(neighbor_index[0])[0],3])
        index=0
        for j in neighbor_index[0]:
            neighbor[index] = cloud_points[j]
            index+=1			

        eigenvalues, eigenvectors = PCA(neighbor)
        all_eigenvalues[i]=eigenvalues
        all_eigenvectors[i]=eigenvectors		

    return all_eigenvalues, all_eigenvectors


def compute_features(cloud_points, radius):

    # Compute the features for all query points in the cloud

    verticality = np.zeros((cloud_points.shape[0]))
    linearity = np.zeros((cloud_points.shape[0]))
    planarity = np.zeros((cloud_points.shape[0]))
    sphericity = np.zeros((cloud_points.shape[0]))
    all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud_points,radius)
    #verticality
    linearity = np.ones((cloud_points.shape[0]))-all_eigenvalues[:,1]/all_eigenvalues[:,2]

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
    print("OK")
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
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, 0.3)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		

    # Features computation
    # ********************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute features on the whole cloud
        verticality, linearity, planarity, sphericity = compute_features(cloud, 0.5)

        # Save cloud with normals
        write_ply('../Lille_street_small_features.ply', (cloud, verticality, linearity, planarity, sphericity), ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
