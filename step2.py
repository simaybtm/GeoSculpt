# Experımental code use step1 for now

import step1 
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


# Run step 1 to get the ground points
#TODO  


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

def main():
    print ("Inializing Step 2...\n")
    ordinary_kriging_interpolation (ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
    print(">> Ordinary Kriging interpolation complete.\n")
    
if __name__ == "__main__":
    # Add arguments to the parser
    parser = argparse.ArgumentParser(description='Create a DTM using Ordinary Kriging')
    parser.add_argument('res', type=float, help='Resolution of the DTM')
    parser.add_argument('minx', type=float, help='Minimum x-coordinate of the DTM')
    parser.add_argument('maxx', type=float, help='Maximum x-coordinate of the DTM')
    parser.add_argument('miny', type=float, help='Minimum y-coordinate of the DTM')
    parser.add_argument('maxy', type=float, help='Maximum y-coordinate of the DTM')
    args = parser.parse_args()
    main()
    