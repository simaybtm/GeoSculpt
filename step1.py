
# THIS IS step1.py
# python step1.py 69EZ1_21.LAZ 190250 313225 190550 313525 0.5 5.0 0.2

import numpy as np
import startinpy as st
from pyinterpolate import build_experimental_variogram, build_theoretical_variogram, kriging
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

import laspy

import argparse


import math
import matplotlib.pyplot as plt # testing output
from mpl_toolkits.mplot3d import Axes3D # testing output
from tqdm import tqdm # Load bar

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
epsilon	    used for classifying the points as ground if they are within "epsilon" distance of the cloth in its final position

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
    thinned_pointcloud = pointcloud[::10]  # every 10th point is selected
    print(f" Number of points after thinning: {thinned_pointcloud.shape[0]}")
    return thinned_pointcloud

# Function to get the valid neighbors of a grid point
def get_valid_neighbors(i, j, z_grid):
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check the four direct neighbors (up, down, left, right)
        ni, nj = i + di, j + dj
        if 0 <= ni < z_grid.shape[0] and 0 <= nj < z_grid.shape[1]:  # Ensure the neighbor is within grid bounds
            neighbors.append(z_grid[ni, nj])
    return neighbors

## Function to implement the Cloth Simulation Filter algorithm (CSF)
def cloth_simulation_filter(pointcloud, csf_res, epsilon, max_iterations=100, delta_z_threshold=0.01):
    print("Running Cloth Simulation Filter algorithm...")

    if pointcloud.size == 0:
        print("Empty point cloud after filtering or thinning.")
        return np.array([]), np.array([])

    inverted_pointcloud = np.copy(pointcloud)
    max_z = np.max(pointcloud[:, 2])
    inverted_pointcloud[:, 2] = max_z - pointcloud[:, 2]
    # Check if inversion was successful
    if np.array_equal(inverted_pointcloud, pointcloud):
        print(" Inversion failed. Aborting CSF.")
        return np.array([]), np.array([])
    else:
        print(" Inversion successful.") 
        
    # Use KDTree for efficient nearest neighbor search
    kd_tree = cKDTree(inverted_pointcloud[:, :2])  # Using only X and Y
    
    # Initializing the cloth/grid
    x_min, x_max = np.min(pointcloud[:, 0]), np.max(pointcloud[:, 0]) # Get the min and max x values to create the grid
    y_min, y_max = np.min(pointcloud[:, 1]), np.max(pointcloud[:, 1])
    x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, csf_res), np.arange(y_min, y_max, csf_res)) # Create the grid
    z_grid = np.full(x_grid.shape, max_z + 10) # Initialize the grid with a value higher than the max z value
    
    # Show "grid/cloth" points above the inverted point cloud
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(inverted_pointcloud[:, 0], inverted_pointcloud[:, 1], inverted_pointcloud[:, 2], c='blue', label='Inverted Point Cloud', s=1)
    # ax.scatter(x_grid, y_grid, z_grid, c='red', label='Grid Points', s=1)
    # ax.set_title('Inverted Point Cloud and Grid/Cloth Points')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()
        
    # Simulating the cloth falling process
    iteration = 0
    while iteration < max_iterations:
        max_change = 0
        for i in range(z_grid.shape[0]):
            for j in range(z_grid.shape[1]):
                dist, idx = kd_tree.query([x_grid[i, j], y_grid[i, j]])
                closest_z = max_z - inverted_pointcloud[idx, 2]

                neighbors = get_valid_neighbors(i, j, z_grid)
                if neighbors:
                    avg_neighbor_z = np.mean(neighbors)
                    new_z = np.min([closest_z + epsilon, avg_neighbor_z])
                else:
                    new_z = closest_z + epsilon

                change = np.abs(z_grid[i, j] - new_z)
                z_grid[i, j] = new_z
                max_change = max(max_change, change)

        if max_change <= delta_z_threshold:
            break
        iteration += 1

    # Classify points as ground or non-ground based on their final distance to the cloth
    ground_points = []
    non_ground_points = []
    
    for point in pointcloud:
        x, y, z = point
        grid_x_idx = np.argmin(np.abs(x_grid[0, :] - x))
        grid_y_idx = np.argmin(np.abs(y_grid[:, 0] - y))
        cloth_z = z_grid[grid_y_idx, grid_x_idx]
        
        # Classify points based on their final distance to the cloth
        if np.abs(z - cloth_z) <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)

    ground_points = np.array(ground_points)
    non_ground_points = np.array(non_ground_points)
    
    # Just check the first 10 points for testing
    # for i, point in enumerate(pointcloud[:10]):  
    #     x, y, z = point
    #     grid_x_idx = np.argmin(np.abs(x_grid[0, :] - x))
    #     grid_y_idx = np.argmin(np.abs(y_grid[:, 0] - y))
    #     cloth_z = z_grid[grid_y_idx, grid_x_idx]
    #     distance_to_cloth = np.abs(z - cloth_z)
    #     print(f"Point {i}: Z = {z}, Cloth Z = {cloth_z}, Distance = {distance_to_cloth}, Classified as {'ground' if distance_to_cloth <= epsilon else 'non-ground'}")
        
    # Show cloth points after the simulation
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(inverted_pointcloud[:, 0], inverted_pointcloud[:, 1], inverted_pointcloud[:, 2], c='blue', label='Inverted Point Cloud', s=1)
    # ax.scatter(x_grid, y_grid, z_grid, c='red', label='Grid Points', s=1)
    # ax.set_title('Inverted Point Cloud and Grid/Cloth Points')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()           
    
    # Show number of ground and non-ground points
    print(f" Number of ground points: {ground_points.shape[0]}")
    print(f" Number of non-ground points: {non_ground_points.shape[0]}")


    # for testing reasons thin the ground points (DELETE AFTER TESTING)
    ground_points = ground_points[::10]
    non_ground_points = non_ground_points[::10]
    print(f" Number of ground points after thinning: {ground_points.shape[0]}")
    print(f" Number of non-ground points after thinning: {non_ground_points.shape[0]}")
    
    return ground_points, non_ground_points

## Function to visualize the separation between ground and non-ground points (testing)
def test_ground_non_ground_separation(ground_points, non_ground_points):
    """
    Visualizes the separation between ground and non-ground points.
    
    Parameters:
    - ground_points: np.array, points classified as ground.
    - non_ground_points: np.array, points classified as non-ground.
    """
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(ground_points) > 0:
        ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='green', label='Ground Points', s=1)
    
    if len(non_ground_points) > 0:
        ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], c='red', label='Non-Ground Points', s=1)
    
    ax.set_title('Ground vs Non-Ground Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.show()
    
    # Plot non-ground points and ground points on separate subplots (3D scatter plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})
    ax1.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='green', label='Ground Points', s=1)
    ax1.set_title('Ground Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()      
    ax2.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], c='red', label='Non-Ground Points', s=1)
    ax2.set_title('Non-Ground Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    plt.show()
    
    # Plot non-ground points and ground points on separate subplots (2D scatter plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    ax1.scatter(ground_points[:, 0], ground_points[:, 1], c='green', label='Ground Points', s=1)
    ax1.set_title('Ground Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax2.scatter(non_ground_points[:, 0], non_ground_points[:, 1], c='red', label='Non-Ground Points', s=1)
    ax2.set_title('Non-Ground Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    plt.show()

## Function to create ground.laz file
def save_ground_points_las(ground_points, filename="ground.laz"):
    print("Saving ground points to LAS file...")
    if ground_points.size > 0:
        # Create a new LAS file with laspy
        header = laspy.LasHeader(version="1.4", point_format=2)
        las = laspy.LasData(header)

        # Assign ground points to the LAS file
        las.x = ground_points[:, 0]
        las.y = ground_points[:, 1]
        las.z = ground_points[:, 2]

        # Write the LAS file to disk
        las.write(filename)
    else:
        print("No ground points found after CSF classification.")

## Function to check if Laplace interpolation is working correctly (visualize with matplotlib)
def visualize_laplace(dtm, minx, maxx, miny, maxy, resolution):
    # Generate the X and Y coordinates for the meshgrid
    x_range = np.arange(minx, maxx, resolution)
    y_range = np.arange(miny, maxy, resolution)
    X, Y = np.meshgrid(x_range, y_range[:dtm.shape[0]])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, dtm, cmap='terrain', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('DTM of Laplace')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
## Function to apply a median filter to the DTM to reduce spikes
def apply_median_filter(dtm, size=3):
    """
    Apply a median filter to the DTM to reduce spikes.
    
    Parameters:
    - dtm: The digital terrain model array.
    - size: The size of the neighborhood to consider for the median filter.
    
    Returns:
    - Filtered DTM.
    """
    return median_filter(dtm, size=size)
 
## Function to create a continuous DTM using Laplace interpolation
def laplace_interpolation(ground_points, resolution, minx, maxx, miny, maxy):
    print("Creating DTM with simplified Laplace interpolation...")

    # Create a grid covering the extents of the ground points with the specified resolution
    x_range = np.arange(minx, maxx, resolution)
    y_range = np.arange(miny, maxy, resolution)
    dtm = np.full((len(y_range), len(x_range)), np.nan)  # Initialize DTM with NaNs

    # Populate the grid with Z-values from ground points (use KDTree for efficient nearest neighbor search)
    kdtree = cKDTree(ground_points[:, :2])
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            _, idx = kdtree.query([x, y], k=1)
            dtm[i, j] = ground_points[idx, 2]      
        
            
    # Supress warning: RuntimeWarning: Mean of empty slice
    np.warnings.filterwarnings('ignore')

    # Perform a simple "smoothing" by averaging non-NaN neighbors
    for i in range(1, dtm.shape[0] - 1):
        for j in range(1, dtm.shape[1] - 1):
            if np.isnan(dtm[i, j]):
                neighbors = dtm[i-1:i+2, j-1:j+2]
                dtm[i, j] = np.nanmean(neighbors)  # Mean of non-NaN neighbors

    # Handle edge cases and improve interpolation quality by filling NaNs with the mean of the entire grid
    dtm = np.where(np.isnan(dtm), np.nanmean(dtm), dtm)

    # Save DTM to a TIFF file
    with rasterio.open('dtm.tiff', 'w', driver='GTiff', height=dtm.shape[0], width=dtm.shape[1], count=1, dtype=str(dtm.dtype), crs='EPSG:32633', transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution)) as dst:
        dst.write(dtm, 1)
    print("DTM saved as dtm.tiff")

    return dtm 

### Step 2: Ordinary Kriging
## Function to check if Ordinary Kriging is working correctly (visualize with matplotlib)
def visualize_ok(dtm, x_range, y_range):
    X, Y = np.meshgrid(x_range, y_range)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')  # Correct way to add 3D axes
    surf = ax.plot_surface(X, Y, dtm, cmap='terrain', linewidth=0, antialiased=False)
    
    plt.title('Digital Terrain Model (DTM) Interpolated with Ordinary Kriging')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Elevation')
    
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
## Function to create a continuous DTM using Ordinary Kriging
def ordinary_kriging_interpolation(ground_points, resolution, minx, maxx, miny, maxy):
    print("Starting Ordinary Kriging interpolation...")

    # Calculate variance of the dataset 
    #variance = np.var(ground_points[:, 2])
    #print(f"Variance of the dataset: {variance}")

    # Prepare the data
    point_data = np.array(ground_points)
    # Thin further for experimental semivariogram
    point_data = point_data[::10]

    # Step 2: Calculate the experimental semivariogram
    search_radius = resolution * 20
    max_range = (maxx - minx) / 2
    
    try: 
        experimental_semivariogram = build_experimental_variogram(input_array=point_data,
                                                                step_size=search_radius,
                                                                max_range=max_range)
        print("Experimental semivariogram calculated.")
        print("EXPERIMENTAL\n",experimental_semivariogram)
    except MemoryError as e:
        print(f"MemoryError: {e}")
        return None
    # Plot experimental semivariogram
    experimental_semivariogram.plot()

    # Step 3: Fit a theoretical semivariogram model
    semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                          model_name='linear', 
                                          sill=30, # Units: meters 
                                          rang=150, # Units: meters
                                          nugget=0)  # Units: meters
    print("\n\nTheoretical semivariogram model fitted.")
    print("\nTHEORETICA\n",semivar)
    
    # Step 4: Perform Ordinary Kriging
    x_coords = np.arange(minx, maxx + resolution, resolution)
    y_coords = np.arange(miny, maxy + resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    unknown_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Predictions
    predictions = kriging(observations=point_data, theoretical_model=semivar,
                          points=unknown_points, how='ok', no_neighbors=32)

    # Reshape predictions to match the grid
    predicted_values = np.array([pred[0] for pred in predictions])
    dtm = predicted_values.reshape(grid_y.shape)

    # Save the DTM to a TIFF file
    transform = from_origin(minx, maxy, resolution, -resolution)
    with rasterio.open('dtm_ordinary_kriging.tiff', 'w', driver='GTiff',
                       height=dtm.shape[0], width=dtm.shape[1],
                       count=1, dtype=str(dtm.dtype), crs='EPSG:4326',
                       transform=transform) as dst:
        dst.write(dtm, 1)

    print("\nDTM saved as dtm_ordinary_kriging.tiff")
    
    # Visualize the DTM created with Ordinary Kriging
    visualize_ok(dtm, x_coords, y_coords)
    
    return dtm

## Main function
def main():
    # Use parsed arguments directly
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon}\n")
   
    # Processing pipeline for Step 1
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        return
    if pointcloud is not None:
        print(">> Point cloud read successfully.\n")
        thinned_pc = thin_pc(pointcloud)
        print(">> Point cloud thinned.\n")
        ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon)
        print (">> Ground points classified with CSF algorithm.\n")
        print("Starting testing of ground and non-ground points separation...")
        test_ground_non_ground_separation(ground_points, non_ground_points)
        print(">> Testing complete.\n")
        # Save the ground points in a file called ground.laz
        save_ground_points_las(ground_points)
        print(">> Ground points saved to ground.laz.\n")
        dtm = laplace_interpolation(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
        print(">> Laplace interpolation complete.\n")
        
        # Apply median filter to the DTM to reduce spikes
        # dtm_filtered = apply_median_filter(dtm, size=5)
        # Visualize or save the filtered DTM
        #visualize_laplace(dtm_filtered, args.minx, args.maxx, args.miny, args.maxy, args.res)
        visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)
        
        # if DTM is saved, print message
        if dtm is not None:
            print(">> DTM saved to output file location.\n")
        
        print("\nStep 1 completed!\n\n")   
    
    # Continue to Step 2    
    print ("Inializing Step 2...\n")
    ordinary_kriging_interpolation (ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
    print(">> Ordinary Kriging interpolation complete.\n")
    

if __name__ == "__main__":
    main()
    