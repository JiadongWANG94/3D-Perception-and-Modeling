#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
from sklearn.preprocessing import label_binarize

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


def cloud_decimation(points, colors, labels, factor):

    decimated_points = points[::factor, :]
    decimated_colors = colors[::factor, :]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels



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



def grid_subsampling_colors(points, colors, voxel_size):

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
    # Values : np array containning [num_points, sum_x, sum_y, sum_z, sum_R, sum_G, sum_B]
    grid_dict = {}

    # Create array from which we will sum values in the dict
    # every line of the array contain : [1, x, y, z, R, G, B]
    data = np.hstack((np.ones((points.shape[0], 1)), points.astype(np.float32), colors.astype(np.float32)))

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
    subsampled_points = subsampled_data[:, 1:4]
    subsampled_colors = subsampled_data[:, 4:].astype(np.uint8)

    return subsampled_points, subsampled_colors



def grid_subsampling_labels(points, colors, labels, voxel_size):

    # Compute voxel indice for each point
    grid_indices = (np.floor(points / voxel_size)).astype(int)

    # Limits of the grid
    min_grid_indices = np.amin(grid_indices, axis=0)
    max_grid_indices = np.amax(grid_indices, axis=0)

    # Number of cells in each direction
    deltaX, deltaY, deltaZ = max_grid_indices - min_grid_indices + 1

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * deltaX + grid_indices[:, 2] * deltaX * deltaY

    #
    # BONUS : Method to choose the label of voxels
    #   In each voxel, count how many times each label value occurs.
    #   => Keep a vector of size n where n is the number of labels. 
    #

    label_values = np.unique(labels)

    # Create array from which we will sum values in the dict
    # every line of the array contain : [1, x, y, z, R, G, B, L_1, ..., L_n]
    boolean_labels = label_binarize(labels, classes=label_values)
    data = np.hstack((np.ones((points.shape[0], 1)), points.astype(np.float32), colors.astype(np.float32), boolean_labels))

    # Create a dictionary
    # Key : vectorized indices
    # Values : np array containning [num_points, sum_x, sum_y, sum_z, sum_R, sum_G, sum_B, , sumL_1, ..., sumL_n]
    grid_dict = {}

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


    # Convert to numpy array
    subsampled_data = np.array([d / d[0] for key, d in grid_dict.items()])

    # Get the predominant labels
    subsampled_labels = label_values[np.argmax(subsampled_data[:, 7:], axis=1)]

    # Get points and colors barycenters
    subsampled_points = subsampled_data[:, 1:4] 
    subsampled_colors = subsampled_data[:, 4:7].astype(np.uint8)

    return subsampled_points, subsampled_colors, subsampled_labels








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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']


    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])



    # Subsample the point cloud on a grid
    # ***********************************
    #
    #


    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling_labels(points, colors, labels, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../grid_subsampled.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    print('Done')
    