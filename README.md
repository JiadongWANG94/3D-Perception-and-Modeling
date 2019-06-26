# 3D Perception and Modeling


A big part of this repository is a python toolbox to process 3D point clouds, including:

```
ToolBox
│   utlis.py
│   │   read_ply
|   └───write_ply
│
└───neighborhoods.py
│   │   brute_force_spherical
│   │   brute_force_KNN
│   └───KDTree
│   
└───subsampling.py
│   │   cloud_decimation
│   │   grid_subsampling
│   │   grid_subsampling_colors
│   └───grid_subsampling_labels
│   
└───transformation.py
│   │   rotation
│   └───translation
│
└───ICP.py
│   │   best_rigid_transform
│   │   icp_point_to_point
│   │   icp_point_to_point_fast
│   └───RMS10_estimation
│
└───descriptors.py
│   │   rot_3D
│   │   PCA
│   │   compute_local_PCA
│   └───compute_features
│
└───RANSAC.py
│   │   compute_plane
│   │   in_plane
│   │   RANSAC
│   └───recursive_RANSAC
│
└───RegionGrowing.py
│   │   compute_curvatures_and_normals
│   │   region_criterion
│   │   queue_criterion
│   │   RegionGrowing
│   └───recursive_RegionGrowing
│
└───reconstruction.py
    │   compute_hoppe
    └───compute_eimls
```
The code of this toolbox is written by *Jean-Emmanuel DESCHAUD*, *Hugues THOMAS* and *Xavier ROYNARD*. This toolbox is offered to Master students of SJTU-ParisTech Elite Institut of Technoligy as correlation of in-class TDs of the course **'Perception 3D et Modélisation'**.

Based on the toolbox, some of applications are done by *Jiadong WANG*:  
**Urban accessibility diagnosis based on MLS point clouds**

This is the final project of class *3D perception and modelizaiton* of SJTU-ParisTech Elite Institut of Technology held by Prof. Jean-Emmanuel Deschaud form Mines ParisTech.  
In this project, I tried to realize part of functions described in the paper *Urban accessibility diagnosis based on MLS point clouds*, including segmentation of facades, roads, curbs and objects.  
Some common technics of point cloud processing are used like cleaning and subsampling.
The code are based on some tools offered by Prof. Jean-Emmanuel Deschaud.

The work flow of the algorithm can be resumed as:
- 0: Pre-processing of data  
- 0.1: Load Original Point Cloud  
- 0.2: Grid Sub-sampling  
- 0.3: Data cleaning  
- 0.3.1: Eliminate isolated points  
- 0.3.2: Eliminate too high points  
- 0.3.3: Set the lowest level of points  
- 1: Load pre-processed Point Cloud  
- 2: Transform the 3D points into range image  
- 3: Fill the holes with morphological transformation  
- 4: Segmentation of ground  
- 5: Segmentation of facades  
- 6: Segmentation of object  
- 7: Segmentation of curbs  
- 7.1: Find original curbs  
- 7.2: Connect directly the curbs too closed  
- 7.3: Use bezier curbs to connect curbs not that closed  
- 8: Color the cloud points  

![Original data](https://github.com/JiadongWANG94/3D-Perception-and-Modelization/blob/master/figures/Data_Set.png)

![Range Image](https://github.com/JiadongWANG94/3D-Perception-and-Modelization/blob/master/figures/Max_Range_Image_Zoom.jpg)

![Output data](https://github.com/JiadongWANG94/3D-Perception-and-Modelization/blob/master/figures/Colored.png)

You can download the data from [IQmulus & TerraMobilita Contest HomePage](http://data.ign.fr/benchmarks/UrbanAnalysis/).

**Suture of MLS point clouds**  
Given a set of sequential frames of MLS point clouds, an algorithm was designed to suture them together. ICP, PCA and regional croissance methods are included.

**3D Reconstruction and Mapping of an Indoor Environment** 
A project of an internship at BOSCH.
