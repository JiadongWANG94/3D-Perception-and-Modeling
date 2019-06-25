#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Plane detection by RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_curvatures_and_normals(points, search_tree, radius):
    
    curvatures = np.zeros(len(points))
    normals = np.zeros((len(points),3))
    
    # TODO:
    for i in range(points.shape[0])
        index_neighbor = search_tree.query_radius()
        neighbors = points[index_neighbor]
        center = np.mean(neighbors, axis=0)
        neigh_c = neighbors - center
        C = (neigh_c.T).dot(neigh_c) / neighbors.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        curvatures[i] = eigenvalues[0]/(np.sum(eigenvalues))
        normals[i] = eigenvectors[2]
 		
    return curvatures, normals 


def region_criterion(p1,p2,n1,n2):
    
    # TODO:
    
    return True

def queue_criterion(c):
    
    # TODO:
    
    return True


def RegionGrowing(cloud, normals, curvatures, search_tree, radius, region_criterion, queue_criterion):
        
    region = np.zeros(len(cloud), dtype=bool)
        
    # TODO:
                
    return region


def recursive_RegionGrowing(cloud, normals, curvatures, search_tree, radius, region_criterion, queue_criterion, NB_PLANES=2):

    plane_inds = np.zeros(0, dtype=int)
    remaining_inds = np.zeros(len(cloud), dtype=int)
    plane_labels = np.zeros(0, dtype=int)
    
    # TODO:
        
    return plane_inds, remaining_inds, plane_labels

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan_sub2cm.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    N = len(points)
    

    # Computes Normals of the whole cloud
    # ************************
    #

    print('\n--- 5) ---\n')
    
    # Build a search tree
    t0 = time.time()
    search_tree = KDTree(points, leaf_size=100)
    t1 = time.time()
    print('KDTree computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Parameters for normals computation
    radius = 0.3
    
    # Computes normals of the whole cloud
    t0 = time.time()    
    curvatures, normals = compute_curvatures_and_normals(points, search_tree, radius)
    t1 = time.time()
    print('normals and curvatures computation done in {:.3f} seconds'.format(t1 - t0))
    

    # Find a plane by Region Growing
    # ***********************************
    #
    #
    
    print('\n--- 6), 7) and 8) ---\n')

    # Define parameters of Region Growing
    radius = 0.5
    
    # Find a plane by Region Growing
    t0 = time.time()
    region = RegionGrowing(points, normals, curvatures, search_tree, radius, region_criterion, queue_criterion)
    t1 = time.time()
    print('Region Growing done in {:.3f} seconds'.format(t1 - t0))
    
    #
    plane_inds = region.nonzero()[0]
    remaining_inds = (1-region).nonzero()[0]
    
    # Save the best plane
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], curvatures[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'curvatures'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds], curvatures[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'curvatures'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 9) ---\n')
    
    # Define parameters of recursive_RANSAC
    radius = 0.3
    NB_PLANES = 50
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RegionGrowing(points, normals, curvatures, search_tree, radius, region_criterion, queue_criterion, NB_PLANES)
    t1 = time.time()
    print('recursive RegionGrowing done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
        
    print('Done')
    