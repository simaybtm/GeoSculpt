
# THIS IS step1.py
# python step1.py 69EZ1_21.LAZ 190250 313225 190550 313525 0.5 5.0 0.2

import numpy as np
import startinpy as st
import pyinterpolate as pi
import rasterio
import scipy
import laspy
import fiona
import shapely
import argparse


import math
import matplotlib.pyplot as plt # testing CSV output
from mpl_toolkits.mplot3d import Axes3D # testing CSV output

'''
INPUTS:
name	    step1.py
inputfile	LAZ
minx	    minx of the bbox for which we want to obtain the DTM
miny	    miny of the bbox for which we want to obtain the DTM
maxx	    maxx of the bbox for which we want to obtain the DTM
maxy	    maxy of the bbox for which we want to obtain the DTM
res	        DTM resolution in meters
csf_res	    resolution in meters for the CSF grid to use
epsilon	    threshold in meters to classify the ground

OUTPUTS:
dtm.tiff    representing the 50cm-resolution DTM of the area created with Laplace
ground.laz  the ground points
'''

# Argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Ground filtering and DTM creation with Laplace.')
parser.add_argument('inputfile', type=str, help='Input LAZ file')
parser.add_argument('minx', type=float, help='Minimum X of the bbox')
parser.add_argument('miny', type=float, help='Minimum Y of the bbox')
parser.add_argument('maxx', type=float, help='Maximum X of the bbox')
parser.add_argument('maxy', type=float, help='Maximum Y of the bbox')
parser.add_argument('res', type=float, help='DTM resolution in meters')
parser.add_argument('csf_res', type=float, help='Resolution in meters for the CSF grid')
parser.add_argument('epsilon', type=float, help='Threshold in meters to classify the ground')
args = parser.parse_args()


'''
TODO
You need to implement the Cloth Simulation Filter algorithm (CSF) as explained in the terrainbook.

You need to classify the points as ground and use those to generate the DTM with Laplace interpolation. 
You are not allowed to use startinpy’s interpolate() function, you need to implement it yourself 
(but you can use startinpy triangulation and other functions).

'''

### Step 1: Ground filtering + DTM creation with Laplace
## Read LAS file with laspy and filter points by bounds
def read_las(file_path, min_x, max_x, min_y, max_y):
    try:
        with laspy.open(file_path) as f:
            las = f.read()
            mask = (las.x >= min_x) & (las.x <= max_x) & (las.y >= min_y) & (las.y <= max_y)
            filtered_points = np.column_stack((las.x[mask], las.y[mask], las.z[mask]))
            print(f"Number of points after applying bounding box filter: {filtered_points.shape[0]}")
            return filtered_points
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

## Function to thin the point cloud by half
def thin_pc(pointcloud):
    print("Thinning the point cloud...")
    thinned_pointcloud = pointcloud[::10] 
    print(f"Number of points after thinning: {thinned_pointcloud.shape[0]}")
    return thinned_pointcloud

## Function to implement the Cloth Simulation Filter algorithm (CSF)
def cloth_simulation_filter(pointcloud, csf_res, epsilon):
    print("Running Cloth Simulation Filter algorithm...")
    
    # Check if the point cloud is empty after filtering/thinning
    if pointcloud.size == 0:
        print("Empty point cloud after filtering or thinning. Please check the bounding box coordinates and thinning process.")
        return np.array([]), np.array([])
    
    # Invert the point cloud (flip Z values)
    inverted_pointcloud = np.copy(pointcloud)
    max_z = np.max(pointcloud[:, 2]) if pointcloud.size > 0 else 0
    inverted_pointcloud[:, 2] = max_z - pointcloud[:, 2]
    
    # Initialize cloth as a grid at a height above the highest point (z0)
    ''' For simplicity, a grid covering the extents of the point cloud with resolution csf_res is created. 
    This grid represents the cloth. The height of the cloth is set to be 10 units above the highest point in the point cloud.'''
    x_min, x_max = np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0])
    y_min, y_max = np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1])
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, csf_res), np.arange(y_min, y_max, csf_res))
    z_grid = np.full(x_grid.shape, max_z + 10)  # Start the cloth 10 units above the highest point
    
    # Simplified simulation of the cloth falling process
    # For each grid point, move it downwards until it is within epsilon of a point in the inverted cloud
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            cloth_point = np.array([x_grid[i, j], y_grid[i, j], z_grid[i, j]])
            # Find the closest point below the cloth point
            distances = np.sqrt(np.sum((inverted_pointcloud[:, :2] - cloth_point[:2])**2, axis=1))
            closest_point_idx = np.argmin(distances)
            closest_point_z = inverted_pointcloud[closest_point_idx, 2]

            # Move the cloth point down if it's above the closest point + epsilon
            if cloth_point[2] > closest_point_z + epsilon:
                z_grid[i, j] = closest_point_z + epsilon

    # Convert the cloth grid back to match the original point cloud orientation
    z_grid = max_z - z_grid

    # Classify points as ground or non-ground based on their distance to the cloth surface
    ground_points = []
    non_ground_points = []
    for point in pointcloud:
        x_idx = np.argmin(np.abs(x_grid[0, :] - point[0]))
        y_idx = np.argmin(np.abs(y_grid[:, 0] - point[1]))
        if np.abs(z_grid[y_idx, x_idx] - point[2]) <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)
            
    '''      
    # Test CSV output with matplotlib
    # Convert lists to numpy arrays for easier handling
    ground_points = np.array(ground_points)
    non_ground_points = np.array(non_ground_points)
    
    # Visualization with matplotlib
    fig = plt.figure(figsize=(20, 10))
    
    # Original Point Cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], s=1, c='k')
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Ground Points
    ax2 = fig.add_subplot(132, projection='3d')
    if ground_points.size > 0:
        ax2.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], s=1, c='g')
    ax2.set_title('Ground Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Non-Ground Points
    ax3 = fig.add_subplot(133, projection='3d')
    if non_ground_points.size > 0:
        ax3.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], s=1, c='r')
    ax3.set_title('Non-Ground Points')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

    return ground_points, non_ground_points
    '''

    return np.array(ground_points), np.array(non_ground_points)
    
## Function to create a continuous DTM using Laplace interpolation
#TODO



## Main function
def main():
    # Use parsed arguments directly
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon}")
   
    # Processing pipeline
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        return
    if pointcloud is not None:
        print("Point cloud read successfully.")
        thinned_pc = thin_pc(pointcloud)
        print("Point cloud thinned.")
        ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon)
        print ("Ground points classified with CSF algorithm.")
        
        # TODO: Implement ground filtering and save ground points as ground.laz
        # TODO: Implement Laplace interpolation with thinned_pc and save DTM as dtm.tiff
        #ground_points = ground_filter_tin(thinned_pc, resolution, d_max, alpha_max)
        #save_ground_points_to_las(ground_points, output_file_path)

        print("Step 1 complete.")   
        

if __name__ == "__main__":
    main()
    