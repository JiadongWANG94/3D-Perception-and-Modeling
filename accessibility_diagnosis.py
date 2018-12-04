#
#
#      0==================================================================0
#      | Project Urban accessibility diagnosis from mobile laser scanning |
#      0==================================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jiadong WANG - 24/10/2018
#
print('importing liberaries: ')
# Import numpy package and name it "np"
import numpy as np

import math 

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time

from skimage import measure
from skimage.morphology import reconstruction
from skimage import morphology

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import axes3d

from PIL import Image

from pylab import *


#import tqdm


######################################################
## Functions

#################################################################
## Sub sampling #################################################

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

#################################################################
## Data Cleaning ################################################

def eliminate_isolate_points(points, radius = 1.0 , n_neighbors =15):
    tree = KDTree(points)
    i = 0
    n_temp = points.shape[0]
    while i<points.shape[0]:
        neighbor = tree.query_radius([points[i]],radius, count_only=True)
        if neighbor[0] < n_neighbors:
            points = np.delete(points, [i],axis=0)
            i-=1
        i+=1		
        if np.mod(i,1000)==0: print('elimination: %f percent' %(i/points.shape[0]*100))
    print('%d points were eliminated.' %(n_temp - points.shape[0]))
    return points

def eliminate_isolate_points_2(points, radius = 2.0 , n_neighbors =15):
    tree = KDTree(points)
    i = 0
    n_temp = points.shape[0]
    index = np.array([True]*points.shape[0])
    for (i,p_temp) in enumerate(points):
        if index[i]:
            neighbor = tree.query_radius([points[i]],r=radius) #, count_only=True
            #print(neighbor)
            if len(neighbor[0]) < n_neighbors:
                index[i]=False	
                index[neighbor[0]]=False
        if np.mod(i,1000)==0: print('elimination: %f percent' %(i/n_temp*100))
    print('%d points were eliminated.' %(n_temp - np.count_nonzero(index)))
    return points[index]

def eliminate_high_points(points, height =5):
    z_min = np.min(points, axis=0)[2]
    i = 0
    n_temp = points.shape[0]
    while i<points.shape[0]:
        if points[i][2] > z_min+height:
            points = np.delete(points, [i],axis=0)
            i-=1
        i+=1		
        if np.mod(i,1000)==0: print('elimination: %f percent' %(i/points.shape[0]*100))
    print('%d points were eliminated.' %(n_temp - points.shape[0]))
    return points

def eliminate_high_points_2(points, height =5):
    z_min = np.min(points, axis=0)[2]
    i = 0
    n_temp = points.shape[0]
    index = points[:,2]<=height+z_min
    points = points[index]
    print('%d points were eliminated.' %(n_temp - points.shape[0]))
    return points

def put_on_the_ground(points, z_min = 0.1):
    z_min_old = np.min(points, axis=0)[2]
    points[:,2] =points[:,2] - z_min_old + z_min 
    return points

################################################################
## Range Image Creation ########################################
def create_range_image(points, pixel_size=0.1):

    """
    \
	value =0 : if no data
	
	
    """
    lowest_point = points[0]

    # Compute voxel indice for each point
    grid_indices = (np.floor(points[:,0:2] / pixel_size)).astype(int)
    

    min_grid_indices = np.min(grid_indices, axis=0)
    max_grid_indices = np.max(grid_indices, axis=0)    

    grid_indices = grid_indices - min_grid_indices
    deltaX, deltaY = max_grid_indices-min_grid_indices + 1
    
    min_range_image = np.zeros((deltaX,deltaY))
    max_range_image = np.zeros((deltaX,deltaY))

    # Scalar equivalent to grid indices
    scalar_indices = grid_indices[:, 0] + grid_indices[:, 1] * deltaX
    
    # put the points in the dictionary of vertical voxels so that after the segmentation we can reconstruct the 3D PC according to the range image
    grid_dict={}
    
    for i,p in enumerate(points):
        if scalar_indices[i] in grid_dict:
            grid_dict[scalar_indices[i]] +=[points[i]]
        else:
            grid_dict[scalar_indices[i]] = [points[i]]

    for i in grid_dict.keys():
        list_points = np.reshape(grid_dict[i],[-1,3])
        z_max = np.max(list_points[:,2])
        z_min = np.min(list_points[:,2])
        min_range_image[np.mod(i,deltaX),i//deltaX] = z_min
        max_range_image[np.mod(i,deltaX),i//deltaX] = z_max

    return min_range_image, max_range_image, deltaX, deltaY, grid_dict, grid_indices

	

################################################################
## Filling holes with morphological reconstruction #############
def fill_holes(image):
    
    marker = np.copy(image)
    max_temp = np.max(image)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            if (marker[i,j]==0): marker[i,j]=max_temp
    return reconstruction(marker, image, method = 'erosion')


#################################################################
## Segmentation of Ground  ######################################
def detect_groud(range_image, remain_indice, threshold = 0.2):
    len_max = 0
    list_max = {(0,0)}
    print('range_max = ', np.max(range_image))
    print('range_min = ', np.min(range_image))
    for i in range(range_image.shape[0]):
        for j in range(range_image.shape[1]):
            if (range_image[i,j]!=0) and remain_indice[i,j]:
                list_1 = {tuple([i,j])}
                queue_1 = {tuple([i,j])}
                while len(queue_1)>0:
                    #print(queue_1)
                    temp = queue_1.pop()
                    #print(temp)
                    p_mid = list(temp)
                    if p_mid[0]+1 < range_image.shape[0] and np.abs(range_image[p_mid[0]+1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])<threshold and (remain_indice[p_mid[0]+1,p_mid[1]]):
                        list_1.add((p_mid[0]+1,p_mid[1]))					
                        queue_1.add((p_mid[0]+1,p_mid[1]))
                        remain_indice[p_mid[0]+1,p_mid[1]]=False
                    if p_mid[0]-1 >= 0 and np.abs(range_image[p_mid[0]-1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])<threshold and (remain_indice[p_mid[0]-1,p_mid[1]]):
                        list_1.add((p_mid[0]-1,p_mid[1]))					
                        queue_1.add((p_mid[0]-1,p_mid[1]))	   
                        remain_indice[p_mid[0]-1,p_mid[1]]=False
                    if p_mid[1]+1 < range_image.shape[1] and np.abs(range_image[p_mid[0],p_mid[1]+1]-range_image[p_mid[0],p_mid[1]])<threshold and (remain_indice[p_mid[0],p_mid[1]+1]):
                        list_1.add((p_mid[0],p_mid[1]+1))					
                        queue_1.add((p_mid[0],p_mid[1]+1))	   
                        remain_indice[p_mid[0],p_mid[1]+1]=False
                    if p_mid[1]-1 >= 0 and np.abs(range_image[p_mid[0],p_mid[1]-1]-range_image[p_mid[0],p_mid[1]])<threshold and (remain_indice[p_mid[0],p_mid[1]-1]):
                        list_1.add((p_mid[0],p_mid[1]-1))					
                        queue_1.add((p_mid[0],p_mid[1]-1))
                        remain_indice[p_mid[0],p_mid[1]-1]=False	
                if len(list_1)>len_max:
                    len_max = len(list_1)
                    list_max = list_1.copy()
                    print('change list, length of new list:  %d' %len(list_1))
    print('Number of Pixels of the set of Ground: ',len_max)
    label_ground = np.zeros(range_image.shape,dtype='bool')
    while (len(list_max)>0):
        p_temp = list_max.pop()
        label_ground[p_temp[0],p_temp[1]]=True
    return label_ground
					
#################################################################
## Segmentation of facades ######################################

def neighbors_in_image_4(range_image, p_mid, exclude):
    neighbors ={tuple(p_mid)}
    if p_mid[0]+1 < range_image.shape[0] and np.abs(range_image[p_mid[0]+1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]+1,p_mid[1]]):
        neighbors.add((p_mid[0]+1,p_mid[1]))					
    if p_mid[0]-1 >= 0 and np.abs(range_image[p_mid[0]-1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]-1,p_mid[1]]):
        neighbors.add((p_mid[0]-1,p_mid[1]))					
    if p_mid[1]+1 < range_image.shape[1] and np.abs(range_image[p_mid[0],p_mid[1]+1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0],p_mid[1]+1]):
        neighbors.add((p_mid[0],p_mid[1]+1))					
    if p_mid[1]-1 >= 0 and np.abs(range_image[p_mid[0],p_mid[1]-1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0],p_mid[1]-1]):
        neighbors.add((p_mid[0],p_mid[1]-1))					
    neighbors.remove(tuple(p_mid))

    return neighbors

def geodesic_length_size_4(range_image, point_1, exclude):
    # similar to BFS
    size = 1
    geodesic = 0
    queue_1 = [point_1]
    value_1 = [0]
    set_1 = {tuple(point_1)}
    while len(queue_1)>0:
        p_temp =queue_1[0]
        v_temp =value_1[0]
        del queue_1[0]
        del value_1[0]
        neighbors = neighbors_in_image_4(range_image, p_temp, exclude)
        while len(neighbors)>0:
            neig_temp = neighbors.pop()
            if range_image[neig_temp[0],neig_temp[1]]==range_image[point_1[0],point_1[1]]:
                queue_1.append(list(neig_temp))
                exclude[neig_temp[0],neig_temp[1]]=True
                value_1.append(v_temp+1)
                set_1.add(neig_temp)
            if v_temp+1>geodesic: geodesic = v_temp+1
        size+=1
    return geodesic, size, set_1		

def detect_facade(range_image, exclude, cut_line_height=4, threshold = 5):

    candidates = range_image > np.min(range_image)+cut_line_height
    candidates = np.logical_and(candidates, ~exclude) 
    label_facade = np.zeros(range_image.shape, dtype='bool')
    for i in range(range_image.shape[0]):
        for j in range(range_image.shape[1]):
            if candidates[i,j]:
                geodesic, size, set_1 = geodesic_length_size_4(candidates, [i,j], exclude)
                if np.power(geodesic,2)*np.pi/4/size > threshold:
                    while len(set_1)>0:
                        tup_1 = set_1.pop()
                        label_facade[tup_1[0],tup_1[1]]=True

    return label_facade
            
#################################################################
## Segmentation of object  ######################################
def detect_object(range_image, exclude):
    """
    we do a binary morphological reconstruction 
    """
    mask = exclude.astype('int')
    marker = np.ones(range_image.shape)
    marker[:,0] = mask[:,0]
    marker[:,-1] = mask[:,-1]
    marker[0,:] = mask[0,:]
    marker[-1,:] = mask[-1,:]

    return (reconstruction(marker, mask, method = 'erosion')- mask).astype('bool')

    
#################################################################
## Segmentation of curbs ######################################


def neighbors_in_image_8(range_image, p_mid, exclude):
    neighbors ={tuple(p_mid)}
    if p_mid[0]+1 < range_image.shape[0] and np.abs(range_image[p_mid[0]+1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]+1,p_mid[1]]):
        neighbors.add((p_mid[0]+1,p_mid[1]))					
    if p_mid[0]-1 >= 0 and np.abs(range_image[p_mid[0]-1,p_mid[1]]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]-1,p_mid[1]]):
        neighbors.add((p_mid[0]-1,p_mid[1]))					
    if p_mid[1]+1 < range_image.shape[1] and np.abs(range_image[p_mid[0],p_mid[1]+1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0],p_mid[1]+1]):
        neighbors.add((p_mid[0],p_mid[1]+1))					
    if p_mid[1]-1 >= 0 and np.abs(range_image[p_mid[0],p_mid[1]-1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0],p_mid[1]-1]):
        neighbors.add((p_mid[0],p_mid[1]-1))					
    if p_mid[0]+1 < range_image.shape[0] and p_mid[1]+1 < range_image.shape[1] and np.abs(range_image[p_mid[0]+1,p_mid[1]+1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]+1,p_mid[1]+1]):
        neighbors.add((p_mid[0]+1,p_mid[1]+1))					
    if p_mid[0]-1 >= 0 and p_mid[1]+1 < range_image.shape[1] and  np.abs(range_image[p_mid[0]-1,p_mid[1]+1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]-1,p_mid[1]+1]):
        neighbors.add((p_mid[0]-1,p_mid[1]+1))					
    if p_mid[0]+1 < range_image.shape[0] and p_mid[1]-1 >= 0 and np.abs(range_image[p_mid[0]+1,p_mid[1]-1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]+1,p_mid[1]-1]):
        neighbors.add((p_mid[0]+1,p_mid[1]-1))					
    if p_mid[0]-1 >= 0 and p_mid[1]-1 >= 0 and np.abs(range_image[p_mid[0]-1,p_mid[1]-1]-range_image[p_mid[0],p_mid[1]])==0 and (~exclude[p_mid[0]-1,p_mid[1]-1]):
        neighbors.add((p_mid[0]-1,p_mid[1]-1))	    

    neighbors.remove(tuple(p_mid))

    return neighbors

def geodesic_length_size_8(range_image, point_1, exclude):
    # similar to BFS
    size = 1
    geodesic = 0
    queue_1 = [point_1]
    value_1 = [0]
    set_1 = {tuple(point_1)}
    while len(queue_1)>0:
        p_temp =queue_1[0]
        v_temp =value_1[0]
        del queue_1[0]
        del value_1[0]
        neighbors = neighbors_in_image_8(range_image, p_temp, exclude)
        while len(neighbors)>0:
            neig_temp = neighbors.pop()
            if range_image[neig_temp[0],neig_temp[1]]==range_image[point_1[0],point_1[1]]:
                queue_1.append(list(neig_temp))
                exclude[neig_temp[0],neig_temp[1]]=True
                value_1.append(v_temp+1)
                set_1.add(neig_temp)
            if v_temp+1>geodesic: geodesic = v_temp+1
        size+=1
    return geodesic, size, set_1		

def detect_curb(curb_candidate,threshold=5):

    label_curb = np.zeros(curb_candidate.shape, dtype='bool')
    exclude = np.logical_not(curb_candidate)
    set_curbs=[]
    for i in range(curb_candidate.shape[0]):
        for j in range(curb_candidate.shape[1]):
            if curb_candidate[i,j]:
                geodesic, size, set_1 = geodesic_length_size_8(curb_candidate.astype('int'), [i,j], exclude)
                if np.power(geodesic,2)*np.pi/4/size > threshold:
                    set_curbs.append(set_1.copy())
                    while len(set_1)>0:
                        tup_1 = set_1.pop()
                        label_curb[tup_1[0],tup_1[1]]=True

    return label_curb, set_curbs

#################################################################
## Connection of curbs ##########################################

# two step:1. connect the endpoints too close, 2. use bezier curb to connect the endpoints not that close

# step 1
def identify_endpoints(set_curbs, label_curb):
    
    n_curbs = len(set_curbs)
    endpoints = np.zeros([len(set_curbs),4])
    for (i,set_1) in enumerate(set_curbs):
        #the two points that the geodesic distance between them reach the geodesic measurement are the two endpoints
        set_copy=set_1.copy()
        geodesic_max = 0
        for point_1 in set_1:
            exclude=~label_curb
            geodesic = 0
            queue_1 = [point_1]
            value_1 = [0]
            exclude[point_1[0],point_1[1]]=True			
            while len(queue_1)>0:
                p_temp =queue_1[0]
                v_temp =value_1[0]
                del queue_1[0]
                del value_1[0]
                neighbors = neighbors_in_image_8(label_curb, p_temp, exclude)
                while len(neighbors)>0:
                    neig_temp = neighbors.pop()
                    if label_curb[neig_temp[0],neig_temp[1]]:
                        queue_1.append(list(neig_temp))
                        value_1.append(v_temp+1)
                        exclude[neig_temp[0],neig_temp[1]]=True
                    if v_temp+1>geodesic: geodesic = v_temp+1
                    if geodesic>=geodesic_max:
                        geodesic_max=geodesic
                        point_f1=[point_1[0],point_1[1]]
                        point_f2=[neig_temp[0],neig_temp[1]]
        endpoints[i]=point_f1+point_f2
    return endpoints

def pixel_passed_by_line(pixel,a,b,c):
    if (np.dot((pixel+np.array([0.5,0.5])),np.array([a,b]).T)+c)*(np.dot((pixel+np.array([-0.5,-0.5])),np.array([a,b]).T)+c)<=0:
        return True
    elif (np.dot((pixel+np.array([-0.5,0.5])),np.array([a,b]).T)+c)*(np.dot((pixel+np.array([0.5,-0.5])),np.array([a,b]).T)+c)<=0:
        return True
    else:
        return False
		
def connect_close_endpoints(endpoints, label_curb, threshold):
    n_curbs=len(endpoints)
    endpoints_1=np.zeros([n_curbs*2,2])
    endpoints_1[:n_curbs,0:2]=endpoints[:,0:2]
    endpoints_1[n_curbs:2*n_curbs,0:2]=endpoints[:,2:4]
    tree = KDTree(endpoints_1)
    for point_1 in endpoints_1:
        point_2 = endpoints_1[tree.query([point_1],2)[1]][0][1]
        if np.sqrt(np.sum(np.power(np.array(point_1)-np.array(point_2),2)))<=threshold:
            if point_1[0]==point_2[0]:
                b=0
                a=1
                c=-point_1[0]
            elif point_1[1]==point_2[1]:
                b=1
                a=0
                c=-point_1[1]
            else:
                a=1/(point_2[0]-point_1[0])		
                b=-1/(point_2[1]-point_1[1])	
                c=-point_1[0]/(point_2[0]-point_1[0])+point_1[1]/(point_2[1]-point_1[1])						
            for i in range(int(point_1[0]),int(point_2[0])+1):
                for j in range(int(point_1[1]),int(point_2[1])+1):
                    if pixel_passed_by_line([i,j],a,b,c):label_curb[i,j]=True
    return label_curb		

# step 2: bezier curbs
def PCA(points):

    # Compute the barycenter
    center = np.mean(points, axis=0)
    
    # Centered clouds
    points_c = points - center

    # Covariance matrix
    C = (points_c.T).dot(points_c) / points.shape[0]

    # Eigenvalues
    return np.linalg.eigh(C)

def compute_vector(P, list_curbs):
    V=[0,0]
    tree = KDTree(list_curbs)
    neighbors = list_curbs[tree.query_radius([P],4)[0]]
    eigenvalue, eigenvector = PCA(neighbors)
    V = eigenvector[:,1]
    return V

def compute_third_point(P0,P2,V0,V2):
    x0 , y0 = P0
    x2 , y2 = P2
    a0 , b0 = V0
    a2 , b2 = V2
    if np.abs(a0*b2-a2*b0)<0.01 or (a0!=0 and a2!=0 and np.abs(math.atan(b0/a0)-math.atan(b2/a2))<np.pi*15/180) or (a0==0 and a2!=0 and np.abs(math.atan(b2/a2))<np.pi*15/180) or (a2==0 and a0!=0 and np.abs(math.atan(b0/a0))<np.pi*15/180):
        return (x0+x2)/2, (y0+y2)/2
    else:
        y1 = (b0*b2*(x0-x2)+b0*a2*y2-b2*a0*y0)/(-b2*a0+b0*a2)
        if b0 != 0:
            return x0-a0/b0*(y0-y1), y1
        else:
            return x2-a2/b2*(y2-y1), y1		
	
def reconnection_bezier(endpoints, set_curbs_connected, range_image, threshold =50):

    label_curb_artificial = np.zeros(range_image.shape, dtype='bool') 
    # fist, label the endpoints with set of points in the curb
    n_curbs = len(set_curbs_connected)
    index_endpoints = np.zeros(2*n_curbs,dtype='int')
    for i in range(2*n_curbs):
        for (j,set_points_in_curb) in enumerate(set_curbs_connected):
            if (tuple(endpoints[i]) in set_points_in_curb) : 
                index_endpoints[i]=j
    # transform list of tuples into np.array
    list_curbs = [0]*n_curbs
    for i in range(n_curbs):
        list_curbs[i]=np.array(list(set_curbs_connected[i]))
    # loop to connect points with bezier curbs
    for (i,P0) in enumerate(endpoints[:-1]):
        for (j,P2) in enumerate(endpoints[i+1:]):
            if index_endpoints[i]!=index_endpoints[j+i+1] and np.sqrt(np.sum(np.power(np.array(P0)-np.array(P2),2)))<threshold:
                V0 = compute_vector(P0, list_curbs[index_endpoints[i]])
                V2 = compute_vector(P2, list_curbs[index_endpoints[j+i+1]])
                P1 = np.array(compute_third_point(P0,P2,V0,V2))
                for t in np.linspace(0,1,100):
                    B = P0*(1-t)**2 + P1*2*(1-t)*t + P2*t**2
                    if 0<=B[0]<=range_image.shape[0]-1 and 0<=B[1]<=range_image.shape[1]-1:
                        label_curb_artificial[int(round(B[0])),int(round(B[1]))]=True
    return label_curb_artificial
				
# substep 1:

#################################################################
## Color ########################################################
def color_by_label(labels, RGBs, points, colors, grid_indices, pixel_size):
    for (i,point) in enumerate(points):
        if len(RGBs[labels[:,int(grid_indices[i,0]),int(grid_indices[i,1])]==True]) >0:
            colors[i]=RGBs[labels[:,int(grid_indices[i,0]),int(grid_indices[i,1])]==True][0]
    return colors
        

######################################################
## Main Part
'''
The work flow:

0: Pre-processing of data
	0.1: Load Original Point Cloud
	0.2: Grid Sub-sampling
	0.3: Data cleaning
		0.3.1: Eliminate isolated points 
		0.3.2: Eliminate too high points 
		0.3.3: Set the lowest level of points 
1: Load pre-processed Point Cloud
2: Transform the 3D points into range image
3: Fill the holes with morphological transformation
4: Segmentation of ground
5: Segmentation of facades
6: Segmentation of object
7: Segmentation of curbs
	7.1: Find orginal curbs
	7.2: Connect directly the curbs too closed
	7.3: Use bezier curbs too connect curbs not that closed
8: Color the cloud points

'''
				
if __name__ == '__main__':


    # Load Point Cloud
    print('Step 0.1 : Load Point Cloud: ')
    t1 = time.time()
    file_path = '../data/Cassette_GT.ply'
    data = read_ply(file_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))
    
    # Subsampling
    print('Step 0.2 : Subsampling:')
    t1 = time.time()
    print('The initial dataset has %d points' %(points.shape[0]))
    n_temp = points.shape[0]
    points = grid_subsampling(points, 0.05)
    print('The resulting dataset has %d points, leaving %f percent of points ' %(points.shape[0], points.shape[0]/n_temp*100))
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))

    # Cleaning points
    print('Step 0.3 : Data cleaning: ')
    '''
    print('0.3.1 : Eliminate isolated points ')
    t1 = time.time()
    print(np.min(points,axis=0))
    points = eliminate_isolate_points_2(points, radius = 2.0 , n_neighbors =15)
    print(np.min(points,axis=0))
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))
'''

    n_temp = points.shape[0]
    index = points[:,2]>=35.66
    points = points[index]

	
###
    print('before:',points.shape)
    print('0.3.2 : Eliminate too high points ')
    t1 = time.time()
    points = eliminate_high_points_2(points)
    t2 = time.time()
    print('after:',points.shape)
    print('Finished in {:.1f}s'.format(t2 - t1))
    print('0.3.3 : Set the lowest level of points ')
    t1 = time.time()
    points = put_on_the_ground(points)
    print(np.min(points,axis=0))
    write_ply('../Cassette_Clean.ply', points, ['x', 'y', 'z'])
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))
    print('Data cleaning end. ')

    # Load Pre-processed Point Cloud
    print('Step 1: Load Pre-processed Point Cloud')
    file_path = '../Cassette_Clean.ply'
    data = read_ply(file_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.copy(points)
    colors[:,:] = 0

    # Transform the 3D points into range image
    print('Step 2: Transform the 3D points into range image')
    t1 = time.time()
    min_range_image, max_range_image, X_grid, Y_grid, grid_dict, grid_indices = create_range_image(points, pixel_size=0.1) #0.2
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))
	
    # Fill the holes with morphological transformation
    print('Step 3: Fill the holes with morphological transformation')
    t1 = time.time()
    max_range_image_filled = fill_holes(max_range_image)
    min_range_image_filled = fill_holes(min_range_image)
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))

    # Segmentation of ground
    print('Step 4: Detection and isolation of ground')
    t1 = time.time()
    holes = np.reshape(np.array([False]*X_grid*Y_grid),[X_grid,Y_grid])
    for i in range(max_range_image_filled.shape[0]):
        for j in range(max_range_image_filled.shape[1]):
            if max_range_image_filled[i,j]==0: holes[i,j]=True		
    label_ground = detect_groud(max_range_image_filled, ~holes)
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))

    # Segmentation of facades
    print('Step 5: Segmentation of facades')
    t1 = time.time()
    label_facade = detect_facade(max_range_image_filled, np.logical_or(label_ground, holes))
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))

    # Segmentation of obstacles
    print('Step 6: Segmentation of obstacles')
    t1 = time.time()
    label_object = detect_object(min_range_image_filled, np.logical_or(label_ground,label_facade))
    t2 = time.time()
    print('Finished in {:.1f}s'.format(t2 - t1))

    # Segmentation of curbs
    print('Step 7: Segmentation of curbs')
    print('7.1: Find orginal curbs')
    dilation_1 = morphology.dilation(min_range_image_filled, selem = np.array([[1,1,1],[1,1,1],[1,1,1]]))#  [[0,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0]]

    ext_grad_1 = dilation_1 - min_range_image_filled


    grad_on_ground =ext_grad_1*label_ground.astype('int')  #_erosion

    # Detection of curbs
    temp1 = grad_on_ground<=0.2
    temp2 = grad_on_ground>=0.04
    label_curb_candidate = np.logical_and(temp1,temp2)
    
    label_curb, set_curbs = detect_curb(label_curb_candidate, threshold =3)

    # Pre-reconnection of curbs
    print('7.2: Connect directly the curbs too closed')
    print('identification of endpoints begin:')
    endpoints = identify_endpoints(set_curbs, label_curb)

    #to plot
    n_curbs=len(set_curbs)
    index_temp=np.zeros([n_curbs*2,2])
    index_temp[:n_curbs,0:2]=endpoints[:,0:2]
    index_temp[n_curbs:2*n_curbs,0:2]=endpoints[:,2:4]
    index_temp=index_temp.T
    index_temp=index_temp.astype('int')
    label_endpoints = np.zeros(max_range_image.shape,dtype='bool')
    label_endpoints[index_temp[0,:],index_temp[1,:]]=True
    

    label_curb_connected=connect_close_endpoints(endpoints, label_curb, 4)

    # use function detect_curb to transfer the label_curb into set and then use identify_endpoints to get endpoints
    label_curb_connected, set_curbs_connected = detect_curb(label_curb_connected.astype('int'), threshold =0.5)
    endpoints_connected = identify_endpoints(set_curbs_connected, label_curb_connected)

    #to plot
    n_curbs=len(set_curbs_connected)
    index_temp=np.zeros([n_curbs*2,2])
    index_temp[:n_curbs,0:2]=endpoints_connected[:,0:2]
    index_temp[n_curbs:2*n_curbs,0:2]=endpoints_connected[:,2:4]
    index_temp=index_temp.T
    index_temp=index_temp.astype('int')
    label_endpoints_connected = np.zeros(max_range_image.shape,dtype='bool')

    label_endpoints_connected[index_temp[0,:],index_temp[1,:]]=True

    # Reconnection of curbs
    print('7.3: Use bezier curbs too connect curbs not that closed')
    endpoints_1 = np.array(index_temp).T
    label_curb_artificial = reconnection_bezier(endpoints_1, set_curbs_connected, max_range_image, threshold=30)

    # Color the points
    print('Step 8: Color the original point cloud')
    labels =np.array([label_facade,(label_ground & ~label_object & ~label_curb_connected & ~label_curb_artificial &~label_facade),label_object,(label_curb_connected & ~label_curb_artificial), label_curb_artificial])
    RGBs = np.array([[0.4,0.5,0.9],[192,192,192],[255,255,0],[255,0,0],[0,255,0]])
    colors = color_by_label(labels, RGBs, points, colors, grid_indices,0.2)
	
    write_ply('../results_colored_points.ply', (points,colors), ['x', 'y', 'z','R','G','B'])

    