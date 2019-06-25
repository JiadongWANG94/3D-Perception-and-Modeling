#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
from sklearn.preprocessing import label_binarize

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


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor,:]
    decimated_colors = colors[::factor,:]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    # YOUR CODE
    subsampled_points = np.array([])
    dic= {}
    for i in range(np.shape(points)[0]):
        tup1=tuple(points[i]//voxel_size)
        if (tup1 in dic):
            tem = dic.pop(tup1)
            tem= np.array(tem)
            dic.update({tup1:np.append(tem,points[i])})
        else:
            dic.update({tup1:points[i]})
    len_t=len(dic)
    for i in range(len_t):
        lst = dic.popitem()[1]
        leng = np.shape(lst)[0]
        subsampled_points=np.append(subsampled_points,[np.mean(np.reshape(lst,[leng//3,3]),axis=0)])
    return np.reshape(subsampled_points,[len_t,3])


def grid_subsampling_colors(points, colors, voxel_size):

    # YOUR CODE
    temp = np.array([])
    dicp = {}
    for i in range(np.shape(points)[0]):
        tup1=tuple(points[i]//voxel_size)
        if (tup1 in dicp):
            tem = dicp.pop(tup1)
            tem = np.array(tem)
            dicp.update({tup1:np.append(tem,np.append(points[i],colors[i].astype('float_')))})
        else:
            dicp.update({tup1:np.append(points[i],colors[i].astype('float_'))})
    len_t=len(dicp)
    for i in range(len_t):
        lst = dicp.popitem()[1]
        leng = np.shape(lst)[0]
        temp = np.append(temp,[np.mean(np.reshape(lst,[leng//6,6]),axis=0)])
    subsampled_points = np.reshape(temp,[len_t,6])[:,0:3]
    subsampled_colors = np.reshape(temp,[len_t,6])[:,3:6].astype('uint8')
    return subsampled_points, subsampled_colors


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
    write_ply('decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points = grid_subsampling(points, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))
    print(subsampled_points)
    # Save
    write_ply('grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])
	
    # Subsample with color
    t0 = time.time()
    subsampled_points, subsampled_colors = grid_subsampling_colors(points,colors, voxel_size)
    t1 = time.time()
    print('Subsampling with color done in {:.3f} seconds'.format(t1 - t0))
    print(subsampled_points)
    print(subsampled_colors)
    # Save
    write_ply('grid_subsampled_color.ply', [subsampled_points,subsampled_colors], ['x', 'y', 'z','red', 'green', 'blue'])
    
    print('Done')
