
# THIS IS step1.py
# python step1.py 69EZ1_21.LAZ 190250 313225 190550 313525 0.5 5.0 0.2

import numpy as np
import startinpy as st
from pyinterpolate import build_experimental_variogram, build_theoretical_variogram, kriging
import rasterio
from rasterio.transform import from_origin
import scipy
import laspy
import fiona
import shapely
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
    print(f"Number of points after thinning: {thinned_pointcloud.shape[0]}")
    return thinned_pointcloud

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
        print("Inversion failed. Aborting CSF.")
        return np.array([]), np.array([])
    else:
        print("Inversion successful.") 
    
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
        prev_z_grid = np.copy(z_grid)
        max_change = 0
        for i in range(z_grid.shape[0]):
            for j in range(z_grid.shape[1]):
                # Find the closest point in the inverted point cloud to each grid point
                dists = np.sqrt((inverted_pointcloud[:, 0] - x_grid[i, j])**2 + (inverted_pointcloud[:, 1] - y_grid[i, j])**2)
                closest_point_idx = np.argmin(dists)
                closest_point_z = inverted_pointcloud[closest_point_idx, 2]

                # Update z_grid based on the closest point and epsilon
                if closest_point_z + epsilon < z_grid[i, j]:
                    z_grid[i, j] = closest_point_z + epsilon

                # Apply tension by averaging with neighbor grid points to simulate internal forces
                neighbor_indices = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                valid_neighbors = [prev_z_grid[ni, nj] for ni, nj in neighbor_indices if 0 <= ni < z_grid.shape[0] and 0 <= nj < z_grid.shape[1]]
                if valid_neighbors:
                    z_grid[i, j] = np.mean([z_grid[i, j]] + valid_neighbors)
                
                # Calculate change for this grid point and update max_change if necessary
                change = np.abs(prev_z_grid[i, j] - z_grid[i, j])
                max_change = max(max_change, change)

                print(f"Iteration {iteration+1}: Max change = {max_change}")
                iteration += 1

                if max_change <= delta_z_threshold:
                    print("Simulation has stabilized.")
                    break

        # Check for convergence
        if np.max(np.abs(prev_z_grid - z_grid)) < delta_z_threshold:
            break
        iteration += 1

    # Classify points as ground or non-ground
    ground_points = []
    non_ground_points = []
    for point in pointcloud:
        x, y, z = point
        grid_x_idx = np.argmin(np.abs(x_grid[0, :] - x))
        grid_y_idx = np.argmin(np.abs(y_grid[:, 0] - y))
        if np.abs(z_grid[grid_y_idx, grid_x_idx] - z) <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)     
        
    # Show cloth points after the simulation
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inverted_pointcloud[:, 0], inverted_pointcloud[:, 1], inverted_pointcloud[:, 2], c='blue', label='Inverted Point Cloud', s=1)
    ax.scatter(x_grid, y_grid, z_grid, c='red', label='Grid Points', s=1)
    ax.set_title('Inverted Point Cloud and Grid/Cloth Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()           


    return np.array(ground_points), np.array(non_ground_points)

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
    
## Function to create a continuous DTM using Laplace interpolation
def laplace_interpolation(ground_points, resolution, minx, maxx, miny, maxy):
    print("Creating DTM with simplified Laplace interpolation...")

    # Create a grid covering the extents of the ground points with the specified resolution
    x_range = np.arange(minx, maxx, resolution)
    y_range = np.arange(miny, maxy, resolution)
    dtm = np.full((len(y_range), len(x_range)), np.nan)  # Initialize DTM with NaNs

    # Populate the grid with Z-values from ground points
    for point in ground_points:
        x_idx = np.searchsorted(x_range, point[0]) - 1
        y_idx = np.searchsorted(y_range, point[1]) - 1
        if 0 <= x_idx < len(x_range) and 0 <= y_idx < len(y_range):
            dtm[y_idx, x_idx] = point[2]  # Assign Z-value to the closest grid point

    # Perform a simple "smoothing" by averaging non-NaN neighbors
    for i in range(1, dtm.shape[0] - 1):
        for j in range(1, dtm.shape[1] - 1):
            if np.isnan(dtm[i, j]):
                neighbors = dtm[i-1:i+2, j-1:j+2]
                dtm[i, j] = np.nanmean(neighbors)  # Mean of non-NaN neighbors

    # TODO: Handle edge cases and improve interpolation quality

    # Save DTM to a TIFF file
    with rasterio.open('dtm.tiff', 'w', driver='GTiff', height=dtm.shape[0], width=dtm.shape[1], count=1, dtype=str(dtm.dtype), crs='EPSG:32633', transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution)) as dst:
        dst.write(dtm, 1)
    print("DTM saved as dtm.tiff")

    # Testing
    #visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)

    return dtm 

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

    # Step 3: Fit a theoretical semivariogram model
    semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                          model_name='linear', 
                                          sill=30, # Units: meters 
                                          rang=150, # Units: meters
                                          nugget=0)  # Units: meters
    print("\n\nTheoretical semivariogram model fitted.")
    print("THEORETICAL\n",semivar)
    
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
    '''
        # Save the ground points in a file called ground.laz
        save_ground_points_las(ground_points)
        print(">> Ground points saved to ground.laz.\n")
        dtm = laplace_interpolation(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
        print(">> Laplace interpolation complete.\n")

        # if DTM is saved, print message
        if dtm is not None:
            print(">> DTM saved to output file location.\n")
        
        print("\nStep 1 completed!\n\n")   
    
    # Continue to Step 2    
    print ("Inializing Step 2...\n")
    ordinary_kriging_interpolation (ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
    print(">> Ordinary Kriging interpolation complete.\n")
    '''

if __name__ == "__main__":
    main()
    