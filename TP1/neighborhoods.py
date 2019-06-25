#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # YOUR CODE
    N_s = np.shape(supports)[0]
    N_q = np.shape(queries)[0]
    arr=[]
    for i in range(N_q):
        arr=arr+[tuple(queries[i])]
    dic = dict.fromkeys(arr)
    #print(arr)
    #print(dic)
    for i in range(N_q):
        for j in range(N_s):
            d=np.sqrt(np.dot((supports[j]-queries[i]),(supports[j]-queries[i]).T))
            if (d<radius):
                valu_old=dic.pop(tuple(queries[i]))
                #print(valu_old)
                #print(supports[j])
                dic.update({tuple(queries[i]):np.append(valu_old,supports[j])})
    neighborhoods = dic

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = None

    return neighborhoods


class neighborhood_grid():

    def __init__(self, points, voxel_size):

        #
        #   Tips :
        #       > "__init__" method is called when you create an object of this class with the line :
        #         grid = neighborhood_grid(points, voxel_size)
        #       > You need to keep here the variables that you want to use later (in the query_radius method).
        #         Just call them "self.XXX". The "__init__" method does not need to return anything
        #

        # Example : save voxel size for later use
        self.grid_voxel_size = voxel_size

        # YOUR CODE

    def query_radius(self, queries, radius):

        #
        #   Tips :
        #       > To speed up the query, you need to find for each point, the part of the grid where its
        #         neighbors can be.
        #       > Then loop over the cells in this part of the grid to find the real neighbors
        #

        # YOUR CODE
        neighborhoods = None

        return neighborhoods


# ------------------------------------------------------------------------------------------
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
    file_path = 'data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

    # Grid neighborhood verification
    # ******************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius = 0.2
        voxel_size = 0.2

        # Create grid structure
        grid = neighborhood_grid(points, voxel_size)

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Get neighborhoods with the grid
        grid_neighborhoods = grid.query_radius(queries, radius)

        # Get neighborhoods with brute force
        brute_neighborhoods = brute_force_spherical(queries, points, radius)

        # Compare all neighborhoods
        print('\nVerification of grid neighborhoods :')
        for n1, n2 in zip(grid_neighborhoods, brute_neighborhoods):
            if n1.shape[0] != n2.shape[0]:
                print('ERROR FOUND : wrong amount of neighbors')
            else:
                diffs = np.unique(n1, axis=0) - np.unique(n2, axis=0)
                error = np.sum(np.abs(diffs))
                if error > 0:
                    print('ERROR FOUND : wrong neighbors')
                else:
                    print('This neighborhood is good')

    # Grid neighborhood timings
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 10
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        voxel_size = 0.2

        # Create grid structure
        grid = neighborhood_grid(points, voxel_size)

        for radius in radius_values:

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            neighborhoods = grid.query_radius(queries, radius)
            t1 = time.time()        
            print('{:d} spherical neighborhood computed in {:.3f} seconds'.format(num_queries, t1 - t0))

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000

        # YOUR CODE