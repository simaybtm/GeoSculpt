# THIS IS step2.py

# python step2.py 69EZ1_21.LAZ 190250 313225 190550 313525 0.1 5.0 2

from step1 import read_las, thin_pc, get_valid_neighbors, filter_elevation_outliers, cloth_simulation_filter, test_ground_non_ground_separation, save_ground_points_las


import numpy as np
import laspy
from pyinterpolate import build_experimental_variogram, build_theoretical_variogram, kriging
import argparse
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import startinpy as st
import rasterio
from scipy.spatial import cKDTree

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

### Step 1: Ground filtering with Cloth Simulation Filter (CSF)
## Functions used from step1.py

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
def ordinary_kriging_interpolation(ground_points, resolution, minx, maxx, miny, maxy, thinning_factor=10, search_radius_factor=10, max_range_factor=2, no_neighbors=16):
    print("Starting Ordinary Kriging interpolation...")

    # Calculate variance of the dataset 
    #variance = np.var(ground_points[:, 2])
    #print(f"Variance of the dataset: {variance}")

    # Prepare the data
    point_data = np.array(ground_points)
    # Thin further for experimental semivariogram
    point_data = ground_points[::thinning_factor]

    # Step 2: Calculate the experimental semivariogram
    search_radius = resolution * search_radius_factor
    max_range = ((maxx - minx) / max_range_factor, (maxy - miny) / max_range_factor)
    
    try:
        experimental_semivariogram = build_experimental_variogram(
            input_array=point_data, step_size=search_radius, max_range=max(max_range))
        print("Experimental semivariogram calculated.")
        print("EXPERIMENTAL MODEL\n",experimental_semivariogram)
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
    print("\nTHEORETICAL MODEL\n",semivar)
    
    # Step 4: Perform Ordinary Kriging
    x_coords = np.arange(minx, maxx + resolution, resolution) 
    y_coords = np.arange(miny, maxy + resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords) # Create a grid of points
    
    unknown_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T # Reshape the grid to a list of points 

    # Predictions
    predictions = kriging(observations=point_data, theoretical_model=semivar,
                          points=unknown_points, how='ok', no_neighbors=no_neighbors)

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
   
    ## Processing pipeline for Step 1: Ground filtering with CSF
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        return
    if pointcloud is not None:
        print(">> Point cloud read successfully.\n")
        thinned_pc = thin_pc(pointcloud, 10)
        print(">> Point cloud thinned.\n")
    
        ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon)
        print (">> Ground points classified with CSF algorithm.\n")
        
        test_ground_non_ground_separation(ground_points, non_ground_points)
        print(">> Testing complete.\n")
        
        # Save the ground points in a file called ground.laz
        save_ground_points_las(ground_points)
        print(">> Ground points saved to ground.laz.\n")
        
         # Outlier detection and removal
        ground_points = filter_elevation_outliers(ground_points, z_threshold=1.0)        
        print(">> Outliers removed.\n")
        
        ## Processing pipeline for Step 2: Ordinary Kriging
        # Perform Ordinary Kriging to create a continuous DTM
        dtm = ordinary_kriging_interpolation(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
        print(">> Ordinary Kriging interpolation complete.\n")
        
        # if DTM is saved, print message
        if dtm is not None:
            print(">> DTM saved to output file location.\n")
        else:
            print(">> DTM could NOT be saved to output file location. :(\n")
        
        print("\nStep 1 completed!\n\n")   

if __name__ == "__main__":
    main()
    