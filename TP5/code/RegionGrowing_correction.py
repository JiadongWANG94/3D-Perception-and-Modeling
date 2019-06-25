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


def compute_curvatures_and_normals(points, search_tree, radius):
    
    all_eigenvalues, all_eigenvectors = compute_local_PCA(points, radius)
    normals = all_eigenvectors[:,:,0]
    curvatures = all_eigenvalues[:,0] / np.sum(all_eigenvalues, axis=1)
    
    return curvatures, normals 


def region_criterion(p1,p2,n1,n2):
    delta = p2-p1
    delta = delta / np.sqrt(np.sum(np.power(delta,2)))
    return ( np.abs(delta.dot(n1)) < 0.01 ) and (np.abs(n1.dot(n2)) > 0.99)

	
def queue_criterion(c):
    return (c < 0.1)


def RegionGrowing(cloud, normals, curvatures, search_tree, radius, region_criterion, queue_criterion):
 
    N = len(cloud)
    
    # Take randomly one point index
    seed_ind = np.random.randint(0, N, size=1)[0]
    
    # Structure to register already seen points
    already_seen = np.zeros(N, dtype=bool)    
    seed_queue = {seed_ind}
    region = np.zeros(N, dtype=bool)
    
    while len(seed_queue)>0:
        seed_ind = seed_queue.pop()
        region[seed_ind] = True
        
        if not(already_seen[seed_ind]):
            already_seen[seed_ind] = True
            
            neighbor_inds = search_tree.query_radius(points[seed_ind,:].reshape(1, -1), radius)[0]
            
            for i, ind in enumerate(neighbor_inds):
                if not(region[ind]) and region_criterion(cloud[seed_ind],cloud[ind],normals[seed_ind],normals[ind]):
                    region[ind] = True
            
                    #if queue_criterion(curvatures[ind]):
                    seed_queue.add(ind)
                        
        print("\rlen(seed_queue):{:9d}".format(len(seed_queue)), end="")
    print("")
            
    return region


def recursive_RegionGrowing(cloud, normals, curvatures, search_tree, radius, region_criterion, queue_criterion, NB_PLANES=2):

    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,N)
    for label in range(NB_PLANES):
        
        # Compute normals
        search_tree = KDTree(cloud[remaining_inds], leaf_size=100)
        
        # Find best plane by RANSAC
        region = RegionGrowing(cloud[remaining_inds,:], normals[remaining_inds,:], curvatures[remaining_inds], search_tree, radius, region_criterion, queue_criterion)
                
        # Update points     
        plane_inds = np.concatenate((plane_inds, remaining_inds[region.nonzero()[0]]))
        plane_labels = np.concatenate((plane_labels, label + np.zeros(np.sum(region))))
        remaining_inds = remaining_inds[(1-region).nonzero()[0]]
        
        print("\tremain_inds: {} --> label: {}".format(len(remaining_inds), label) )
        
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
    radius = 0.1
    
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
    radius = 0.1
    
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
    radius = 0.1
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
    