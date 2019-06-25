#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      First script of the practical session. Plane detection by RANSAC
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


def compute_plane(points):
    
    ref_point = points[0]
    normal = np.cross(points[1]-points[0], points[2]-points[0])
    normal = normal / np.sqrt(np.sum(np.power(normal,2)))
    return ( ref_point, normal )



def in_plane(points, ref_pt, normal, threshold_in=0.1):

    return (np.abs(np.dot((points - ref_pt) , normal)) < threshold_in)



def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    
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


def recursive_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):

    N = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,N)
    for label in range(NB_PLANES):
        # Find best plane by RANSAC
        best_ref_pt, best_normal, best_vote = RANSAC(points[remaining_inds], NB_RANDOM_DRAWS, threshold_in)
        
        # Find points in the plane
        is_in_plane = in_plane(points[remaining_inds], best_ref_pt, best_normal, threshold_in)
        
        # Update points     
        plane_inds = np.concatenate((plane_inds, remaining_inds[is_in_plane.nonzero()[0]]))
        plane_labels = np.concatenate((plane_labels, label + np.zeros(np.sum(is_in_plane))))
        remaining_inds = remaining_inds[(1-is_in_plane).nonzero()[0]]
        
        print("\tbest_vote: {} --> remaining_inds: {} --> label: {}".format(best_vote, len(remaining_inds), label) )
        
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
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.05

    # Take randomly three points
    pts = points[np.random.randint(0, N, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    ref_pt, normal = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    NB_RANDOM_DRAWS = 100
    threshold_in = 0.05

    # Find best plane by RANSAC
    t0 = time.time()
    best_ref_pt, best_normal, best_vote = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    NB_RANDOM_DRAWS = 100
    threshold_in = 0.05
    NB_PLANES = 5
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    