#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

import itertools



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    # Initiate neighborhoods list
    neighborhoods = [None for _ in range(queries.shape[0])]

    # Compute neighbors, one query at a time
    for i, q in enumerate(queries):

        # Compute differences
        differences = supports - q

        # Compute square distances
        square_dists = np.sum(np.power(differences, 2), axis=1)

        # Gather neighborhoods
        neighborhoods[i] = supports[square_dists < radius * radius, :]


    return neighborhoods



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




class neighborhood_grid_2():

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
        self.voxels = {}

        for point in points:

            [x, y, z] = point
            (i, j, k) = (int(x / voxel_size), int(y / voxel_size), int(z / voxel_size))

            voxel = self.voxels.get((i, j, k))

            if voxel is None:
                self.voxels[(i, j, k)] = [point]
            else:
                self.voxels[(i, j, k)] += [point]

    def query_radius(self, queries, radius):

        #
        #   Tips :
        #       > To speed up the query, you need to find for each point, the part of the grid where its
        #         neighbors can be.
        #       > Then loop over the cells in this part of the grid to find the real neighbors
        #

        neighborhoods = []

        for q in queries:

            [x, y, z] = q
            (imin, imax) = (int((x - radius) / self.grid_voxel_size), int((x + radius) / self.grid_voxel_size))
            (jmin, jmax) = (int((y - radius) / self.grid_voxel_size), int((y + radius) / self.grid_voxel_size))
            (kmin, kmax) = (int((z - radius) / self.grid_voxel_size), int((z + radius) / self.grid_voxel_size))

            # Candidate voxels where we can find neighbor points
            voxels = [self.voxels.get(t) for t in itertools.product(range(imin, imax + 1), range(jmin, jmax + 1), range(kmin, kmax + 1)) if self.voxels.get(t) is not None]
            # To get the list of the points, we "flatten the list"
            points = np.array([p for v in voxels for p in v])
            # We compute the square distances between the query and candidate points
            sq_dists = np.sum((points - q)**2, axis=-1)
            # We easily get the neighbor points
            neighborhoods += [points[sq_dists < radius**2]]

        return neighborhoods




class neighborhood_grid():

    def __init__(self, points, voxel_size):

        # Save voxel size
        self.voxel_size = voxel_size

        # Compute voxel indice for each point
        grid_indices = (np.floor(points / voxel_size)).astype(int)
        
        # Create a dictionary
        # Key : tuple indices
        # Values : np array containning all points
        self.grid_dict = {}

        # Start loop over the points to fill the dictionaries
        t0 = time.time()
        for i, p in enumerate(points):
            key = tuple(grid_indices[i, :])
            if key in self.grid_dict:
                self.grid_dict[key] = np.vstack((self.grid_dict[key], p))
            else:
                self.grid_dict[key] = p

            # Print progress and time spend
            if (i % 500000 == 0):
                t = time.time()
                print('{:.1f}% done in {:.1f} seconds'.format(100 * i / points.shape[0], t - t0))


    def query_radius(self, queries, radius):

        # Initiate neighborhoods list
        neighborhoods = [None for _ in range(queries.shape[0])]

        # Compute neighbors, one query at a time
        for i, q in enumerate(queries):

            # Find the minimal/maximal grid indices where we are going to search
            x_min, y_min, z_min = (np.floor((q - radius) / self.voxel_size)).astype(int)
            x_max, y_max, z_max = (np.floor((q + radius) / self.voxel_size)).astype(int)

            # Initiate neighborhood candidates list
            neighbor_candidates = []

            # Loop over possible voxels
            for x_grid in range(x_min, x_max + 1):
                for y_grid in range(y_min, y_max + 1):
                    for z_grid in range(z_min, z_max + 1):

                        # Tuple the grid indices
                        key = (x_grid, y_grid, z_grid)

                        # Verify if points are present here
                        if self.grid_dict.get(key) is not None:

                            # Add neighborhood candidates to the list
                            neighbor_candidates += [self.grid_dict[key]]

            # Stack neighbors
            neighbor_candidates = np.vstack(neighbor_candidates)

            # Compute differences
            differences = neighbor_candidates - q

            # Compute square distances
            square_dists = np.sum(np.power(differences, 2), axis=1)

            # Gather neighborhoods
            neighborhoods[i] = neighbor_candidates[square_dists < radius * radius, :]


        return neighborhoods









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
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

   

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

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
        grid = neighborhood_grid_2(points, voxel_size)

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
                print('ERROR FOUND : wrong ammount of neighbors')
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
    if True:

        # Define the search parameters
        num_queries = 100
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

        for voxel_size in [0.2, 1]:

            # Create grid structure
            print('\nBuilding grid with voxel_size = {:.2f}'.format(voxel_size))
            grid = neighborhood_grid(points, voxel_size)

            print('\n radius (m) |  computing 10 queries (sec)  |')
            print('{:-^12s}|{:-^30s}|'.format('', '', ''))

            for radius in radius_values:

                # Pick random queries
                random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
                queries = points[random_indices, :]

                # Search spherical
                t0 = time.time()
                neighborhoods = grid.query_radius(queries, radius)
                t1 = time.time()        
                print('{:^12.2f}|{:^30.3f}|'.format(radius, t1 - t0))


        print('\nBrute force method')
        print('\n radius (m) |  computing 10 queries (sec)  |')
        print('{:-^12s}|{:-^30s}|'.format('', '', ''))

        for radius in radius_values:

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            neighborhoods = brute_force_spherical(queries, points, radius)
            t1 = time.time()
            print('{:^12.2f}|{:^30.3f}|'.format(radius, t1 - t0))



    # KDTree neighborhoods
    # ********************
    #


    # If statement to skip this part if wanted
    if False:

        print('\n--- Question 5 ---\n')

        # 10000 queries for good approximation of mean time
        num_queries = 1000

        # We will try 3 different radius values
        radius_values = [0.2, 1, 2]

        # Evaluate the influence of leaf size on timing
        lf_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

        print('|     leaf size     | building tree (sec) | 20 cm queries (sec) |  1 m queries (sec)  |  2 m queries (sec)  |')
        print('|{:-^19s}|{:-^21s}|{:-^21s}|{:-^21s}|{:-^21s}|'.format('', '', '', '', ''))
        for lf in lf_values:

            # Build tree
            t0 = time.time()
            tree = KDTree(points, leaf_size=lf)
            t1 = time.time()

            t = []
            for radius in radius_values:

                # Pick random queries
                random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
                queries = points[random_indices, :]

                # Search spherical neighbors
                neighborhoods = None
                t += [time.time()]
                neighborhoods = tree.query_radius(queries, radius)
                t += [time.time()]

            print('|{:^19d}|{:^21.3f}|{:^21.3f}|{:^21.3f}|{:^21.3f}|'.format(lf, t1 - t0, t[1] - t[0], t[3] - t[2], t[5] - t[4]))



    # If statement to skip this part if wanted
    if False:

        print('\n--- Question 6 ---\n')

        # Define the search parameters
        num_queries = 1000
        radius_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

        t0 = time.time()
        tree = KDTree(points, leaf_size=100)
        t1 = time.time()
        print('KDTree built in {:.6f} seconds\n'.format(t1 - t0))


        print(' radius (m) | computing 1000 queries (sec) | computing whole cloud  (sec) |')
        print('{:-^12s}|{:-^30s}|{:-^30s}|'.format('', '', ''))
        for radius in radius_values:

            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]

            # Search spherical
            t0 = time.time()
            neighborhoods = tree.query_radius(queries, radius)
            t1 = time.time()

            # Time to compute all neighborhoods in the cloud
            total_spherical_time = points.shape[0] * (t1 - t0) / num_queries

            print('{:^12.1f}|{:^30.3f}|{:^30.3f}|'.format(radius, t1 - t0, total_spherical_time))
