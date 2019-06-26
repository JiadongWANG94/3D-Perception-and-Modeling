#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#




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
    #   Load the file '../data/bunny.ply'
    #   (See read_ply function)
    #


    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    density = data['scalar_density']



    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #


    # Find the centroid of the cloud
    centroid = np.mean(points, axis=0)

    # Center the cloud on its centroid
    points = points - centroid

    # Divide the scale by 2
    points = points * 0.5

    # Define the rotation matrix (-90° rotation around z-axis)
    angle = -np.pi/2
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

    # Apply the rotation
    points = points.dot(R)

    # Recenter the cloud    
    points = points + centroid

    # Apply last translation
    points = points + np.array([0, -0.1, 0])



    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #


    # Save point cloud
    data = write_ply('../little_bunny', [points, colors, density], ['x', 'y', 'z', 'red', 'green', 'blue', 'density'])

    print('Done')
    