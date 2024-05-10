# python step2.py 69GN2_14.LAZ 198350 308950 198600 309200 1 2 0.5

from step1 import read_las, thin_pc, cloth_simulation_filter, remove_outliers_with_tin, test_ground_non_ground_separation, save_ground_points_las, knn_outlier_removal


import numpy as np
import laspy
from pyinterpolate import build_experimental_variogram, build_theoretical_variogram, kriging

import argparse
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from tqdm import tqdm 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, dtm, cmap='terrain', linewidth=0, antialiased=False)
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
##  Divides the dataset into a training set and a test set.
def create_train_test(dataset, training_set_ratio=0.3, seed=101):
    """
    Args:
    dataset (np.ndarray): The complete dataset array where each row is a data point.
    training_set_ratio (float): The proportion of the dataset to include in the train split.
    seed (int): Seed for the random number generator for reproducibility.

    Returns:
    tuple: Two numpy arrays, (training_set, test_set)
    """
    np.random.seed(seed)
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    split = int(training_set_ratio * len(indices))
    training_idx, testing_idx = indices[:split], indices[split:]
    
    return dataset[training_idx], dataset[testing_idx]

## Function to create a continuous DTM using Ordinary Kriging
def ordinary_kriging_interpolation(ground_points, resolution, minx, maxx, miny, maxy, thinning_factor=20, step_size=3, max_range_factor=0.8, no_neighbors=8):
    print(" Thinning factor:", thinning_factor, "step_size:", step_size, "max_range_factor:", max_range_factor, "no_neighbors:", no_neighbors)

    # Calculate variance of the dataset 
    variance = np.var(ground_points[:, 2])
    print(f" Variance of the dataset: {variance}\n")

    # Prepare the data
    point_data = np.array(ground_points)
    # Thin further for experimental semivariogram
    point_data = ground_points[::thinning_factor]
    print(f" Points before thinning: {len(ground_points)} | Points after thinning: {len(point_data)}\n")

    # Set the search radius and max range
    search_radius = resolution * step_size 
    max_range = ((maxx - minx) / max_range_factor, (maxy - miny) / max_range_factor) 
    print(f" Max range: {max_range}\n")
    
    # Creating train and test datasets
    train_set, test_set = create_train_test(ground_points)
    
    # Step 2: Calculate the experimental semivariogram
    print(" Calculating the experimental semivariogram...")
    try:
        experimental_semivariogram = build_experimental_variogram(
            input_array=point_data, step_size=search_radius, max_range=max(max_range))
        print("Experimental semivariogram calculated.")
        print("EXPERIMENTAL MODEL\n",experimental_semivariogram)
    except MemoryError as e:
        print(f"\nMemoryError: {e}\n")
        sys.exit(1) # Exit program
    
    # Plot variogram
    experimental_semivariogram.plot()
    
    # Step 3: Fit a theoretical semivariogram model
    print("\n Fitting a theoretical semivariogram model...")
    semivar = build_theoretical_variogram(experimental_variogram=experimental_semivariogram,
                                          model_name='linear', 
                                          sill=variance, # meters 
                                          rang=300, # meters 
                                          nugget=0)  # meters
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

    # Reshape predictions to match the grid -> kriging returns a list of tuples 
    predicted_values = np.array([pred[0] for pred in predictions])
    dtm = predicted_values.reshape(grid_y.shape)

    # Save the DTM to a TIFF file
    transform = from_origin(minx, maxy, resolution, -resolution)
    with rasterio.open('dtm_ordinary_kriging.tiff', 'w', driver='GTiff',
                       height=dtm.shape[0], width=dtm.shape[1],
                       count=1, dtype=str(dtm.dtype), crs='EPSG:28992', # Dutch National Grid (Amersfoort / RD New)--> EPSG:28992
                       transform=transform) as dst:
        dst.write(dtm, 1)

    
    # Visualize the DTM created with Ordinary Kriging
    visualize_ok(dtm, x_coords, y_coords)
    
    return dtm

## Function to compute Jackknife RMSE for Ordinary Kriging    
def jackknife_rmse_ok(ground_points, resolution, minx, maxx, miny, maxy, sample_size=1000, thinning_factor=10, search_radius_factor=1, max_range_factor=2, no_neighbors=8):
    if len(ground_points) > sample_size:
        # Randomly sample points if the dataset is larger than the sample size
        indices = np.random.choice(len(ground_points), sample_size, replace=False)
        sample_points = ground_points[indices]
    else:
        sample_points = ground_points

    errors = []
    n = len(sample_points)
    for i in tqdm(range(n), desc="\nComputing Jackknife RMSE for Ordinary Kriging"):
        # Exclude the current point
        subset_points = np.delete(sample_points, i, axis=0)
        
        # Re-run your Ordinary Kriging interpolation on the subset
        dtm = ordinary_kriging_interpolation(subset_points, resolution, minx, maxx, miny, maxy, thinning_factor, search_radius_factor, max_range_factor, no_neighbors)
        
        omitted_point = sample_points[i]
        # Calculate the grid index of the omitted point
        grid_x_idx = int((omitted_point[0] - minx) / resolution)
        grid_y_idx = int((omitted_point[1] - miny) / resolution)
        if 0 <= grid_x_idx < dtm.shape[1] and 0 <= grid_y_idx < dtm.shape[0]:
            z_estimated = dtm[grid_y_idx, grid_x_idx]
            if np.isfinite(z_estimated):
                errors.append((z_estimated - omitted_point[2]) ** 2)

    rmse = np.sqrt(np.mean(errors))
    return rmse

## Main function
def main():
    """
    # Use parsed arguments directly
    print("\n")
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, \
maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon} \n")

    #------ Step 1: Ground filtering with CSF ------
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        # Terminating the program if no points are found
        return
    else:
        print(">> Point cloud read successfully.\n")

    # 1. Thinning
    thinned_pc = thin_pc(pointcloud, 50) # thinning percentage (0-100)
    print(">> Point cloud thinned.\n")
    # 2. Outlier removal
    thinned_pc = knn_outlier_removal(thinned_pc, 10)  # k value for k-NN outlier removal
    print(">> Outliers removed succesfully.\n")
    # 3. Ground filtering with CSF
    ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon, args.minx, args.maxx, args.miny, args.maxy)
    print ("\n>> Ground points classified with CSF algorithm.\n")
    # ADDITIONAL: Testing ground and non-ground points
    #test_ground_non_ground_separation(ground_points, non_ground_points)
    #print(">> Testing ground and non-ground points complete.\n")
    
    # ADDITIONAL: Remove stubborn outliers with TIN
    print("Removing stubborn outliers with TIN...")
    ground_points = remove_outliers_with_tin(ground_points)
    """
    #------ Step 2: Ordinary Kriging ------
    # Cheat here if you already ran step 1 and have the ground points saved ("new_ground_p.las")
    print(" Cheating here, using the ground points from step 1 to skip CSF computation again...\n")
    print( " Warning: The same parameters used in step 1 should be used here as well. \n")
    ground_points = read_las("ground.laz", args.minx, args.maxx, args.miny, args.maxy)
    
    if ground_points is None or ground_points.size == 0:
        print("No ground points found within the specified bounding box.")
        # Terminating the program if no points are found
        return
    

    # ADDITIONAL: Jackknife RMSE computation for Ordinary Kriging
    #print("Starting Jackknife RMSE computation for Ordinary Kriging...")
    #jackknife_rmse_value = jackknife_rmse_ok(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy, sample_size=1000)  # Sample size of 1000 for illustration
    #print(f"Jackknife RMSE for Ordinary Kriging: {jackknife_rmse_value}")

    # 4. Perform Ordinary Kriging to create a continuous DTM
    print("\nStarting Ordinary Kriging interpolation...")
    dtm = ordinary_kriging_interpolation(ground_points, args.res, args.minx, args.maxx, args.miny, args.maxy)
    print("\nDTM saved as dtm_ordinary_kriging.tiff")
    print(">> Ordinary Kriging interpolation complete.\n")

    # 5. Save the ground points in a file called ground.laz
    save_ground_points_las(ground_points)
    print(">> Ground points saved to ground.laz.\n")
        
    print("\nStep 2 completed!\n\n")   

if __name__ == "__main__":
    main()
    