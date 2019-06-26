#
#
#      0===========================================================0
#      |              TP6 SLAM                                     |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jiadong WANG - 24/10/2018
#

# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import tqdm


######################################################
## Functions

## Part 1 sub sampling #########################################

def cloud_decimation(points, factor):

    decimated_points = points[::factor, :]
    #decimated_colors = colors[::factor, :]
    #decimated_labels = labels[::factor]

    return decimated_points


def grid_subsampling(points, voxel_size):

    # Compute voxel indice for each point
    grid_indices = (np.floor(points / voxel_size)).astype(int)

    # Limits of the grid
    min_grid_indices = np.amin(grid_indices, axis=0)
    max_grid_indices = np.amax(grid_indices, axis=0)

    # Number of cells in each direction
    deltaX, deltaY, deltaZ = max_grid_indices - min_grid_indices + 1

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * deltaX + grid_indices[:, 2] * deltaX * deltaY
    
    # Create a dictionary
    # Key : vectorized indices
    # Values : np array containning [num_points, sum_x, sum_y, sum_z]
    grid_dict = {}

    # Create array from which we will sum values in the dict
    # every line of the array contain : [1, x, y, z]
    data = np.hstack((np.ones((points.shape[0], 1)), points.astype(np.float32)))

    # Start loop over the points to fill the dictionaries
    t0 = time.time()
    for i, d in enumerate(data):
        if scalar_indices[i] in grid_dict:
            grid_dict[scalar_indices[i]] += d
        else:
            grid_dict[scalar_indices[i]] = d

        # Print progress and time spend
        if (i % 100000 == 0):
            t = time.time()
            print('{:.1f}% done in {:.1f} seconds'.format(100 * i / points.shape[0], t - t0))


    # Divide by the number of point per cell and convert to numpy array
    subsampled_data = np.array([d / d[0] for key, d in grid_dict.items()])
    subsampled_points = subsampled_data[:, 1:]

    return subsampled_points




## Part 2 RANSAC #########################################
def compute_plane(points):
    
    ref_point = points[0]
    normal = np.cross(points[1]-points[0], points[2]-points[0])
    normal = normal / np.sqrt(np.sum(np.power(normal,2)))
    return ( ref_point, normal )



def in_plane(points, ref_pt, normal, threshold_in=0.2):

    return (np.abs(np.dot((points - ref_pt) , normal)) < threshold_in)



def RANSAC(points, NB_RANDOM_DRAWS=150, threshold_in=0.2):
    
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


## Part 3 Regional Croissance #########################################


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
            print(seed_ind) #Jiadong
            neighbor_inds = search_tree.query_radius(cloud[seed_ind,:].reshape(1, -1), radius)[0]
            
            for i, ind in enumerate(neighbor_inds):
                if not(region[ind]) and region_criterion(cloud[seed_ind],cloud[ind],normals[seed_ind],normals[ind]):
                    region[ind] = True
            
                    #if queue_criterion(curvatures[ind]):
                    seed_queue.add(ind)
                        
        print("\rlen(seed_queue):{:9d}".format(len(seed_queue)), end="")
    print("")
            
    return region

def elimination_small_objects(points,x_max,y_max,z_max):
    
    # Build a search tree
    search_tree = KDTree(points, leaf_size=100)

    # Parameters for normals computation
    radius = 0.1

    # Computes normals of the whole cloud
    curvatures, normals = compute_curvatures_and_normals(points, search_tree, radius)

    # Define parameters of Region Growing
    radius = 0.1

    # Iteration to eliminate the small regions
    N_elimination = 200
    
    Indice = [True]*len(points)
    for i in range(N_elimination):
        region = RegionGrowing(points, normals, curvatures, search_tree, radius, region_criterion, queue_criterion)
        points_in_region = points[region]
        points_max = np.max(points_in_region,axis=0)
        points_min = np.min(points_in_region,axis=0)
        size_of_region = points_max - points_min
        if (size_of_region[0]<=x_max)or(size_of_region[1]<=y_max)or(size_of_region[2]<=z_max):
            Indice = Indice & ~region
    points = points[Indice]

    return points

    




## Part 4 ICP #########################################


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
         ref = (d x N) matrix where "N" is the number of point and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # Barycenters
    ref_center = np.mean(ref, axis=1).reshape((-1, 1))
    data_center = np.mean(data, axis=1).reshape((-1, 1))
    
    # Centered clouds
    ref_c = ref - ref_center
    data_c = data - data_center

    # H matrix
    H = data_c.dot(ref_c.T)

    # SVD on H
    U, S, Vt = np.linalg.svd(H)

    # Checking R determinant
    if np.linalg.det((Vt.T).dot(U.T)) < 0:
        U[:, -1] *= -1

    # Getting R and T
    R = (Vt.T).dot(U.T)
    T = ref_center - R.dot(data_center)

    return R, T



def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    search_tree = KDTree(ref.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    for i in range(max_iter):

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned.T, return_distance=True)

        # Compute average distance
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        R, T = best_rigid_transform(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T


    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, num_queries, final_overlap):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
        num_queries = number of random point used to optimize transformation
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
           
    '''

    # Check
    assert(final_overlap <= 1)

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    t1 = time.time()
    search_tree = KDTree(ref.T)
    t2 = time.time()
    print('KDTree constructed in {:.1f}s'.format(t2 - t1))

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # New num_queries considering that we are going to erase points
    num_queries_0 = int(np.floor(num_queries / final_overlap))


    for i in range(max_iter):

        # Pick random queries
        data_indices = np.random.choice(data_aligned.shape[1], num_queries_0, replace=False)

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned[:, data_indices].T, return_distance=True)

        # Erase the farthest points
        if final_overlap < 1:

            # Find the order sorting neighbors by distance
            sorting_order = np.argsort(distances, axis=0).ravel()

            # Erase farthest neighbors
            distances = distances[sorting_order[:num_queries]]
            indices = indices[sorting_order[:num_queries]]
            data_indices = data_indices[sorting_order[:num_queries]]

        # Compute average distance
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        R, T = best_rigid_transform(data[:, data_indices], ref[:, indices.ravel()])        

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T
        print(i, RMS)


    return data_aligned, R_list, T_list, neighbors_list, RMS_list














######################################################
## Main Part
				
if __name__ == '__main__':

    N_files = 50
	
    # Initiaion of Data Set
    data = list(range(N_files))
    #data = np.astype('string')
    points = list(range(N_files))



    # Load Point Cloud
    print('Load Point Cloud begin: ')
    for i in range(N_files):
        if i<10:
            file_path = '../data/frames/frame_00000%d.ply'%i
        else:
            file_path = '../data/frames/frame_0000%d.ply'%i
        data[i] = read_ply(file_path)
        points[i] = np.vstack((data[i]['x'], data[i]['y'], data[i]['z'])).T
    print('Load Point Cloud end. ')


    # Sub Sampling
    print('Sub-Sampling begin: ')
    """
    Decimation_factor = 50
    for i in range(N_files):
        points[i] = cloud_decimation(points[i],Decimation_factor) 
    """
    voxel_size = 0.2
    for i in range(N_files):
        points[i] = grid_subsampling(points[i],voxel_size) 	
    print('Sub-Sampling end. ')

    # Extraction of ground by RANSAC
    print('RANSAC begin: ')
    for i in range(N_files):
        print(i)
        best_ref_pt, best_normal, best_vote = RANSAC(points[i])
        points[i] = points[i][~in_plane(points[i], best_ref_pt, best_normal, threshold_in=0.2)]
    print('RANSAC end. ')

    # Extraction of dynamic objects by Regional Croissance
    print('Extraction begin: ')
    for i in range(N_files):
        points[i] = elimination_small_objects(points[i], 4,4,2)
        print(i,'//%d iterations' %N_files)        
    print('Extraction end. ')

    # ICP 
    print('ICP begin: ')
    data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(points[1].T,points[0].T,20,1e-4)
    points[1] = data_aligned.T
    for j in range(2,N_files):  
        points[j] = (np.dot(R_list[-1],points[j].T)+T_list[-1]).T
    for i in range(2,N_files):
        data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(points[i].T,points[i-1].T,20,1e-4)
        points[i] = data_aligned.T
        for j in range(i+1,N_files):  
            points[j] = (np.dot(R_list[-1],points[j].T)+T_list[-1]).T
        print(i,'//%d iterations' %N_files, ' RMS = ', RMS_list[-1])
    print(RMS_list)


    points_f = np.array([])
    for i in range(1,N_files):
        points_f=np.append(points_f, points[i])	

    points_f = np.reshape(points_f,[-1,3])
    print('ICP end. ')   


    # Save the results of ICP
    write_ply('../results_Jiadong.ply', points_f, ['x', 'y', 'z'])
    write_ply('../results_Jiadong_testP0.ply', points[0], ['x', 'y', 'z'])
    write_ply('../results_Jiadong_testP1.ply', points[1], ['x', 'y', 'z'])
    write_ply('../results_Jiadong_testP30.ply', points[30], ['x', 'y', 'z'])
    write_ply('../results_Jiadong_testP31.ply', points[31], ['x', 'y', 'z'])
    