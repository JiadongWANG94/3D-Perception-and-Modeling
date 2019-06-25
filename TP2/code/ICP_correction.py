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
#      Hugues THOMAS - 13/12/2017
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



def RMS10_estimation(data, ref):

    # Create a neighbor structure on ref
    print('constructing KDTree ...')
    search_tree = KDTree(ref.T)

    # Find the nearest neighbors
    print('Finding neighbors ...')
    distances, indices = search_tree.query(data.T, return_distance=True)

    # Sort distances
    distances = np.sort(distances, axis=None)

    # Compute RMS
    N = len(distances)
    RMS10 = np.zeros((10,))
    for i in range(10):
        lim = int(np.floor(N * (i + 1) / 10))
        RMS10[i] = np.sqrt(np.mean(np.power(distances[:lim], 2)))


    return RMS10




#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':


    # Load point clouds
    # *****************
    #


   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if True:

        # Load clouds
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Get average distances
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))


   
    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Load clouds
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)


    # If statement to skip this part if wanted
    if False:

        # Load clouds
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)


   
    # Average distances plots
    # ***********************
    #

    # If statement to skip this part if wanted
    if False:

        # Load clouds
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))  

        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        plt.plot(RMS_list)
        plt.show()


    # If statement to skip this part if wanted
    if False:

        # Load clouds
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        plt.plot(RMS_list)
        plt.show()



    # Average distances plots
    # ***********************
    #

    # If statement to skip this part if wanted
    if False:

        # Load clouds
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'
        NDDC_1_ply = read_ply(NDDC_1_path)
        NDDC_2_ply = read_ply(NDDC_2_path)
        NDDC_1 = np.vstack((NDDC_1_ply['x'], NDDC_1_ply['y'], NDDC_1_ply['z']))
        NDDC_2 = np.vstack((NDDC_2_ply['x'], NDDC_2_ply['y'], NDDC_2_ply['z']))


        NDDC_2_bad, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(NDDC_2, NDDC_1, 50, 1e-4, 10000, 1)
        #plt.plot(RMS_list)
        #plt.show()

        NDDC_2_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(NDDC_2, NDDC_1, 50, 1e-4, 10000, 0.8)
        #plt.plot(RMS_list)
        #plt.show()

        print(NDDC_2_bad.shape)

        write_ply('../NDDC_2_myICP_100.ply',[NDDC_2_bad.T], ['x', 'y', 'z'])
        write_ply('../NDDC_2_myICP_80.ply',[NDDC_2_opt.T], ['x', 'y', 'z'])


        # Path of the files
        NDDC_2_50_path = '../data/Notre_Dame_Des_Champs_2_50.ply'
        NDDC_2_80_path = '../data/Notre_Dame_Des_Champs_2_80.ply'

        # Load point clouds
        NDDC_2_50_ply = read_ply(NDDC_2_50_path)
        NDDC_2_80_ply = read_ply(NDDC_2_80_path)

        # Concatenate data
        NDDC_2_50 = np.vstack((NDDC_2_50_ply['x'], NDDC_2_50_ply['y'], NDDC_2_50_ply['z']))
        NDDC_2_80 = np.vstack((NDDC_2_80_ply['x'], NDDC_2_80_ply['y'], NDDC_2_80_ply['z']))


        RMS10_o = RMS10_estimation(NDDC_2_opt, NDDC_1)
        RMS10_50 = RMS10_estimation(NDDC_2_50, NDDC_1)
        RMS10_80 = RMS10_estimation(NDDC_2_80, NDDC_1)
        print(RMS10_o)
        print(RMS10_50)
        print(RMS10_80)
        plt.plot([(i + 1) * 10 for i in range(10)], RMS10_o, label='mine')
        plt.plot([(i + 1) * 10 for i in range(10)], RMS10_50, label='50')
        plt.plot([(i + 1) * 10 for i in range(10)], RMS10_80, label='80')
        plt.legend()
        plt.show()
