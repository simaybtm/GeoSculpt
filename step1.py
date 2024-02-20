
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
import matplotlib.pyplot as plt # testing CSF output
from mpl_toolkits.mplot3d import Axes3D # testing CSF output
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
        print(f">> Ground points saved as {filename}.\n")
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
    visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)

    return dtm 

## Function to check if Ordinary Kriging is working correctly (visualize with matplotlib)
def visualize_ok(dtm, x_range, y_range):
    X, Y = np.meshgrid(x_range, y_range)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, dtm, cmap='terrain', linewidth=0, antialiased=False)
    plt.title('Digital Terrain Model (DTM) Interpolated with Ordinary Kriging')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Elevation')
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
## Function to create a continuous DTM using Ordinary Kriging
def ordinary_kriging_interpolation(ground_points, resolution, minx, maxx, miny, maxy):
    print("Creating DTM with Ordinary Kriging interpolation...")
    
    '''
    Perform OK on the ground.laz created in the step above, and create a 0.5mX0.5m DTM.
    For this step, you should use Pyinterpolate and document in the report the steps you took 
    and show the results of the intermediate steps (eg the experimental variogram and the nugget/range/sill/function you used).
    For this step, the parameters you find by modelling the dataset can be hardcoded, but these should be 
    documented in the report and you still have to submit the code you code (which is only for this dataset).  
    
    '''
    # Convert ground points numpy array to list of [x, y, value] for pyinterpolate
    point_data = ground_points.tolist()

    # Set parameters for variogram analysis
    search_radius = resolution * 5  # Example: 5 times the resolution of your DTM
    max_range = (maxx - minx) / 2  # Example: half the width of your study area

    # 1. Build experimental variogram
    experimental_semivariogram = build_experimental_variogram(input_array=point_data, step_size=search_radius, max_range=max_range)

    # 2. Fit theoretical variogram model (spherical model used as example)
    semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram, model_type='spherical', sill=400, rang=20000, nugget=0)

    # Simulate progress for the kriging process
    num_points = len(point_data)
    with tqdm(total=num_points, desc="Performing Kriging") as pbar:
        for _ in range(num_points):
            pbar.update(1)  # Update the progress bar 
            
    # 3. Perform Ordinary Kriging
    # Prepare a grid of unknown points
    x_range = np.arange(minx, maxx, resolution)
    y_range = np.arange(miny, maxy, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    unknown_points = np.vstack([X.ravel(), Y.ravel()]).T

    # Convert unknown points to list of (x, y) tuples
    unknown_points_list = unknown_points.tolist()

    # Perform kriging
    predictions = kriging(observations=point_data, theoretical_model=semivar, points=unknown_points_list, how='ok', no_neighbors=32)

    # 4. Reshape predicted values back into grid shape for DTM
    predicted_values = np.array([pred[0] for pred in predictions])  # Extracting predicted values
    dtm = predicted_values.reshape(X.shape)

    # 5. Visualize the DTM (optional)
    visualize_ok(dtm, x_range, y_range)

    # 6. Save the DTM to a TIFF file (optional)
    # Define the raster's metadata
    transform = from_origin(minx, maxy, resolution, resolution)
    nrows, ncols = dtm.shape
    with rasterio.open(
        'dtm_ordinary_kriging.tiff', 'w', driver='GTiff',
        height=nrows, width=ncols,
        count=1, dtype=dtm.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(dtm, 1)

    print("DTM saved as dtm_ordinary_kriging.tiff")
    
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
        # Save the ground points in a file called ground.laz
        save_ground_points_las(ground_points)
        print(">> Ground points saved to ground.laz.\n")
        dtm = laplace_interpolation(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
        print(">> Laplace interpolation complete.\n")
        # if DTM is saved, print message
        if dtm is not None:
            print(">> DTM saved to output file location.\n")
        
        print("\nStep 1 complete!\n\n")   
    
    # Continue to Step 2    
    print ("Inializing Step 2...")
    ordinary_kriging_interpolation (ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
    print(">> Ordinary Kriging interpolation complete.\n")

if __name__ == "__main__":
    main()
    