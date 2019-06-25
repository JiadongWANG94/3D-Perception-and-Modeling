#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros(data.shape[0])
    (d,N) = np.shape(data)
    pm2 = np.mean(data,axis=0)
    pm1 = np.mean(ref,axis=0)
    Q1 = ref - pm1
    Q2 = data - pm2
    H = np.dot(Q2.T,Q1)
    U,S,Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,U.T)
    if (np.linalg.det(R)<0):
        N_temp = np.shape(R)[0]
        I_temp = np.eye(N_temp)
        I_temp[N,temp, N_temp]=-1
        R = np.dot(R,I_temp)
    T = pm1-np.dot(R,pm2)    
    return R, T

def brute_force_KNN(queries, supports, k):

    # Initiate neighborhoods list
    neighborhoods = np.empty((k, 3, queries.shape[0]))

    # Compute neighbors, one query at a time
    for i, q in enumerate(queries):

        # Compute differences
        differences = supports - q

        # Compute square distances
        square_dists = np.sum(np.power(differences, 2), axis=1)

        # Get the indices for the k smallest dists
        # partition is faster than sort because it does not sort all elements, but only separate de k smallest from the rest        
        #neighbors_inds = np.argsort(square_dists)[:k]
        neighbors_inds = np.argpartition(square_dists, k)[:k] 
     
        # Gather neighborhoods
        neighborhoods[:, :, i] = supports[neighbors_inds, :]
    return neighborhoods

def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []

    # YOUR CODE

    return data_aligned, R_list, T_list, neighbors_list


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds
        data_o = read_ply(bunny_o_path)
        ref_o = read_ply(bunny_r_path)
        data = np.vstack((data_o['x'], data_o['y'], data_o['z'])).T		
        ref = np.vstack((ref_o['x'], ref_o['y'], ref_o['z'])).T		
		
        # Find the best transformation
        R, T =best_rigid_transform(data, ref)
        # Apply the tranformation
        data_out = np.dot(R,data.T).T + T
        # Save cloud
        write_ply('../bunny_best_rigid_transformation.ply', [data_out], ['x', 'y', 'z'])
        # Compute RMS
        N_temp = np.shape(data)[0]
        RMS = np.sqrt(1./N_temp*np.sum((data-ref)**2))  
        RMS_out = np.sqrt(1./N_temp*np.sum((data_out-ref)**2))  
        # Print RMS
        print('RMS_initial= ',RMS)
        print('RMS_final= ',RMS_out)		

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds

        # Apply ICP

        # Show ICP

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z'])).T		        
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z'])).T
		
        # Load clouds

        # Apply ICP

        # Show ICP


    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds

        # Apply fast ICP for different values of the sampling_limit parameter

        # Plot RMS
        #
        # => To plot something in python use the function plt.plot() to create the figure and 
        #    then plt.show() to display it
