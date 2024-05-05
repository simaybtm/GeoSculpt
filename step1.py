
# Test run with: python step1.py 69GN2_14.LAZ 198350 308950 198600 309200 8.0 5.0 4


# You have to use the following Python libraries: numpy, startinpy, pyinterpolate, rasterio, scipy, laspy, fiona, shapely.
import numpy as np
import startinpy
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import laspy

import argparse


# LIBRARIES FOR TESTING
import matplotlib.pyplot as plt # testing output
from mpl_toolkits.mplot3d import Axes3D # testing output
from tqdm import tqdm # Loading bar to assess time of execution
from sklearn.metrics import mean_squared_error # testing output



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

### Step 1: Ground filtering with CSF
## Read LAS file with laspy and filter points by bounds
def read_las(file_path, min_x, max_x, min_y, max_y):
    try:
        with laspy.open(file_path) as f:
            las = f.read()
            # Filter points within the specified bounding box
            mask = (las.x >= min_x) & (las.x <= max_x) & (las.y >= min_y) & (las.y <= max_y)
            filtered_points = np.column_stack((las.x[mask], las.y[mask], las.z[mask]))
            print(f"Number of points after applying bounding box filter: {filtered_points.shape[0]}")
            return filtered_points

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

## Function to thin the point cloud by half
def thin_pc(pointcloud, thinning_value):
    print("Thinning the point cloud...")
    # Ensure thinning_value is an integer and greater than 0 to avoid errors
    if not isinstance(thinning_value, int) or thinning_value <= 0:
        raise ValueError("Thinning value must be an integer greater than 0.")
    thinned_pointcloud = pointcloud[::thinning_value]  
    print(f" Number of points before thinning: {pointcloud.shape[0]}")
    print(f" Number of points after thinning: {thinned_pointcloud.shape[0]}")

    # Plot the thinned point cloud in 3D (_1_)
    """
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(thinned_pointcloud[:, 0], thinned_pointcloud[:, 1], thinned_pointcloud[:, 2], c='blue', s=1)
    ax.set_title('Thinned Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show() 
    """
    
    return thinned_pointcloud

# Function to detect and remove outliers using k-Nearest Neighbors
def knn_outlier_removal(thinned_pointcloud, k):
    print("Detecting and removing outliers...")
    # Calculate the technical information
    min_x, max_x = np.min(thinned_pointcloud[:, 0]), np.max(thinned_pointcloud[:, 0])
    min_y, max_y = np.min(thinned_pointcloud[:, 1]), np.max(thinned_pointcloud[:, 1])
    min_z, max_z = np.min(thinned_pointcloud[:, 2]), np.max(thinned_pointcloud[:, 2])

    area_width = max_x - min_x
    area_height = max_y - min_y
    height_difference = max_z - min_z

    print(f"Parcel size: {area_width:.2f} m x {area_height:.2f} m")
    print(f"Height difference from min to max point is {height_difference:.2f} meters")

    # Build KDTree for efficient neighbor search
    tree = cKDTree(thinned_pointcloud[:, :2])  # Use only X, Y for spatial queries

    # Compute the distances to the k-th nearest neighbor
    distances, _ = tree.query(thinned_pointcloud[:, :2], k=k + 1)  # k+1 because the point itself is included
    knn_distances = distances[:, k]  # We take the k-th nearest distance

    # Define threshold for outlier detection
    threshold = np.mean(knn_distances) + 2 * np.std(knn_distances)

    # Filter points where the k-th nearest neighbor is within the threshold
    non_outliers = thinned_pointcloud[knn_distances < threshold]

    print(f"Removed {len(thinned_pointcloud) - len(non_outliers)} outliers.")
    return non_outliers

# (USED in CSF) Function to get the valid neighbors of a grid point
def get_valid_neighbors(i, j, z_grid, check_above=False):
    directions = [(-1, 0), (1, 0), (0, -1)] # Left, Right, Down
    if check_above:
        directions.append((0, 1))  
    return [z_grid[i + di, j + dj] for di, dj in directions if 0 <= i + di < z_grid.shape[0] and 0 <= j + dj < z_grid.shape[1]]

## Function to implement the Cloth Simulation Filter algorithm (CSF)
def cloth_simulation_filter(thinned_pointcloud, csf_res, epsilon, max_iterations=None):
    print("Running Cloth Simulation Filter algorithm...")

    if thinned_pointcloud.size == 0:
        print("ERROR: Empty point cloud after filtering or thinning! Aborting CSF.")
        return np.array([]), np.array([])

    max_z = np.max(thinned_pointcloud[:, 2])
    inverted_pointcloud = thinned_pointcloud.copy()
    inverted_pointcloud[:, 2] = max_z - thinned_pointcloud[:, 2]
    print("     Inversion of terrain successful.")

    # Find the highest points in the inverted point cloud to initialize the grid higher than the max z value
    max_z = np.max(inverted_pointcloud[:, 2])
    print(f"        Max Z: {max_z} thus initializing the grid higher than the max z value by 10 units (initial_z = max_z + 10).")
    initial_z = max_z + 10

    # Use KDTree for efficient nearest neighbor search
    kd_tree = cKDTree(inverted_pointcloud[:, :2])

    # Initializing the cloth/grid
    x_grid, y_grid = np.meshgrid(
        np.arange(np.min(thinned_pointcloud[:, 0]), np.max(thinned_pointcloud[:, 0]), csf_res),
        np.arange(np.min(thinned_pointcloud[:, 1]), np.max(thinned_pointcloud[:, 1]), csf_res)
    )
    z_grid = np.full(x_grid.shape, initial_z) 

    # If max_iterations is not set, dynamically compute it as 75% of the total number of thinned points
    if not max_iterations:
        max_iterations = int(0.75 * len(thinned_pointcloud))
        print(f"        Dynamically setting max_iterations to 75% of total thinned points: {max_iterations} / {len(thinned_pointcloud)}")

    # Simulating the cloth falling process
    with tqdm(total=max_iterations, desc="  Cloth Simulation Filter Progress") as pbar:
        for iteration in range(max_iterations):
            for i in range(z_grid.shape[0]):
                for j in range(z_grid.shape[1]):
                    dist, idx = kd_tree.query([x_grid[i, j], y_grid[i, j]], k=1)
                    closest_z = max_z - inverted_pointcloud[idx, 2]
                    neighbors = get_valid_neighbors(i, j, z_grid)
                    if neighbors:
                        z_grid[i, j] = (closest_z + epsilon + np.mean(neighbors)) / 2 # Average of closest_z + epsilon and neighbors
                    else:
                        z_grid[i, j] = closest_z + epsilon # No neighbors, set to closest_z + epsilon
            pbar.update(1)

    # Classifying points as ground or non-ground
    ground_points = []
    non_ground_points = []
    for point in thinned_pointcloud:
        grid_x_idx = int((point[0] - np.min(thinned_pointcloud[:, 0])) / csf_res)
        grid_y_idx = int((point[1] - np.min(thinned_pointcloud[:, 1])) / csf_res)
        cloth_z = z_grid[grid_y_idx, grid_x_idx]
        if np.abs(point[2] - cloth_z) <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)

    ground_points = np.array(ground_points)
    non_ground_points = np.array(non_ground_points)

    ## PLOTS    
    # Show cloth points after the simulation (_2_)
    """ 
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(thinned_pointcloud[:, 0], thinned_pointcloud[:, 1], thinned_pointcloud[:, 2], c='blue', label='Thinned Points', s=1)
    ax.scatter(x_grid, y_grid, z_grid, c='red', label='Cloth Points', s=1)
    ax.set_title('Cloth Points after Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()     
    """
    # Show cloth surface after simulation (_3_)
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, cmap='terrain', edgecolor='none')
    ax.set_title('Cloth Surface after Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    """

    # Show number of ground and non-ground points
    print(f"    Number of ground points: {len(ground_points)}")
    print(f"    Number of non-ground points: {len(non_ground_points)}")

    # Number of shared points (must be zero)
    if len(ground_points.intersection(non_ground_points)) > 0:
        print("    WARNING: Shared points found between ground and non-ground points.")

        shared_points = ground_points.intersection(non_ground_points)
        shared_points = np.array(list(shared_points))
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(shared_points[:, 0], shared_points[:, 1], shared_points[:, 2], c='blue', label='Shared Points', s=1)
        ax.set_title('Shared Points between Ground and Non-Ground Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    else:
        print("    No shared points found between ground and non-ground points!")

    return ground_points, non_ground_points

## (TESTING) Function to visualize the separation between ground and non-ground points
def test_ground_non_ground_separation(ground_points, non_ground_points):
    """
    Visualizes the separation between ground and non-ground points.
    
    Parameters:
    - ground_points: np.array, points classified as ground.
    - non_ground_points: np.array, points classified as non-ground.
    """
    print("Starting testing of ground and non-ground points separation...")

    # STATISTICAL INFORMATION
    # See if there are exreme changes in the Z values of the ground points
    print("     Ground Points - Statistical Information:")
    print(f"        Mean Z: {np.mean(ground_points[:, 2])}")
    print(f"        Median Z: {np.median(ground_points[:, 2])}")
    print(f"        Standard Deviation Z: {np.std(ground_points[:, 2])}")
    print(f"        Min Z: {np.min(ground_points[:, 2])}")
    print(f"        Max Z: {np.max(ground_points[:, 2])}\n")

    # Plot ground points and non-ground points in 3D (green: ground, red: non-ground) (_4_)
    fig = plt.figure(figsize=(15, 10))  
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='green', label='Ground Points', s=1)
    ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2] + 50, c='red', label='Non-Ground Points', s=1)
    ax.set_title('Ground and Non-Ground Points Separation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    
    # Plot non-ground points and ground points on separate subplots (3D) (_5_)
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
    
    # Plot non-ground points and ground points on separate subplots (2D) (_6_)
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
    print("Saving ground points to a new LAS file called ground.laz...")
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

### Step 2: Laplace Interpolation
## (USED in Laplace) Function to compute the Jackknife RMSE for Laplace interpolation
def jackknife_rmse_laplace(ground_points, minx, maxx, miny, maxy, resolution):
    """
    Compute the Jackknife RMSE for Laplace interpolation.

    Input:
        ground_points: np.array, array of ground points with columns [x, y, z].
        minx, maxx, miny, maxy: float, bounding box coordinates.
        resolution: float, resolution of the DTM grid.
    Output:
        rmse: float, the Root Mean Squared Error (RMSE) of the Laplace interpolation.
    """
    errors = []
    n = len(ground_points)
    for i in tqdm(range(n), desc="Computing Jackknife RMSE for Laplace Interpolation"):
        # Exclude the current point
        subset_points = np.delete(ground_points, i, axis=0)
        
        # Re-run your Laplace interpolation on the subset
        dtm = laplace_interpolation(subset_points, minx, maxx, miny, maxy, resolution)
        
        omitted_point = ground_points[i]
        # Estimate z-value at the omitted point's location
            # Assuming the dtm grid aligns exactly with your point locations, which might not always be the case
        grid_x_idx = int((omitted_point[0] - minx) / resolution)
        grid_y_idx = int((omitted_point[1] - miny) / resolution)
        if 0 <= grid_x_idx < dtm.shape[1] and 0 <= grid_y_idx < dtm.shape[0]:  # Check bounds
            z_estimated = dtm[grid_y_idx, grid_x_idx]
            if np.isfinite(z_estimated):  # Ensure estimated value is not NaN
                errors.append((z_estimated - omitted_point[2]) ** 2)

    rmse = np.sqrt(np.mean(errors))
    return rmse

## (TESTING) Function to check if Laplace interpolation is working correctly (visualize with matplotlib)
def visualize_laplace(dtm, minx, maxx, miny, maxy, resolution):
    """
    Visualize the Digital Terrain Model (DTM) created with Laplace interpolation.

    Input:
        dtm: np.array, a 2D array representing the DTM.
        minx, maxx, miny, maxy: float, bounding box coordinates.
        resolution: float, resolution of the DTM grid.
    Output: 
        3D plot of the DTM.
    """
    x_range = np.linspace(minx, maxx, num=dtm.shape[1])
    y_range = np.linspace(miny, maxy, num=dtm.shape[0])
    X, Y = np.meshgrid(x_range, y_range)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, dtm, cmap='terrain', edgecolor='none')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Digital Terrain Model (DTM) Interpolated with Laplace')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Elevation')
    plt.show()

## Function to create a continuous DTM using Laplace interpolation
def laplace_interpolation(ground_points, minx, maxx, miny, maxy, resolution):
    dt = startinpy.DT()
    for pt in ground_points:
        dt.insert_one_pt(pt[0], pt[1], pt[2])
    
    # Preparing grid for interpolation
    x_range = np.linspace(minx, maxx, num=int((maxx - minx) / resolution)) 
    y_range = np.linspace(miny, maxy, num=int((maxy - miny) / resolution))
    dtm = np.full((len(y_range), len(x_range)), np.nan) # Initialize with NaN values

    # Build convex hull (2D) for the ground points and interpolate the z values
    hull = ConvexHull(ground_points[:, :2])
    print("Creating the convex hull (2D) for the ground points to interpolate...")
    # Boundary of the las VS the boundary of the convex hull
    print(" Boundary of the las file: ", minx, maxx, miny, maxy)
    print(" Boundary of the convex hull: ", hull.min_bound, hull.max_bound)
    print("\n")

    # Interpolate the z values for each point in the grid
    for j, y in enumerate(y_range):
        for i, x in enumerate(x_range):
            if point_in_hull((x, y), hull):
                dtm[j, i] = interpolate_z_value(dt, x, y, hull) # Interpolate the z value
            else:
                dtm[j, i] = np.nan  # Points outside the convex hull do NOT get interpolated (assigned NaN)

    # Save the DTM to a TIFF file
    transform = from_origin(minx, maxy, resolution, -resolution)  
    with rasterio.open('dtm_laplace.tiff', 'w', driver='GTiff',
                       height=dtm.shape[0], width=dtm.shape[1],
                       count=1, dtype=str(dtm.dtype), crs='EPSG:28992', # Dutch National Grid (Amersfoort / RD New)
                       transform=transform) as dst:
        dst.write(dtm, 1)
    
    return dtm

# (USED in Laplace) Function to check if a point is inside the convex hull
def point_in_hull(point, hull):
    return all((np.dot(eq[:-1], point) + eq[-1] <= 1e-12) for eq in hull.equations)

## (USED in Laplace) Function to interpolate the z value at a specific point using barycentric coordinates
def interpolate_z_value(dt, x, y, hull):
    if not dt.is_inside_convex_hull(x, y):
        return np.nan  # Outside the convex hull
    triangle = dt.locate(x, y)
    if dt.is_finite(triangle):
        vertices = [dt.get_point(idx) for idx in triangle]
        return weighted_barycentric_interpolate(x, y, vertices, hull)
    return np.nan


## (USED in interpolate_z_value) Function to interpolate the z value using barycentric coordinates
def weighted_barycentric_interpolate(x, y, vertices, hull):
    x1, y1, z1 = vertices[0]
    x2, y2, z2 = vertices[1]
    x3, y3, z3 = vertices[2]

    # Calculate distances from the point to each vertex
    distances = np.array([np.sqrt((x - vx)**2 + (y - vy)**2) for vx, vy, vz in vertices])
    # Normalize distances
    weights = 1 / distances
    if any(np.dot(eq[:-1], (x, y)) + eq[-1] > 0 for eq in hull.equations):  # Check if near the edge
        weights *= 0.5  # Reduce the influence of vertices if near the edge

    # Normalize weights
    weights /= np.sum(weights)

    # Calculate weighted Z using normalized weights
    z = weights[0] * z1 + weights[1] * z2 + weights[2] * z3
    return z

## Main function
def main():
    # Use parsed arguments directly
    print("\n")
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, \
maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon} \n")

    ## Step 1: Ground filtering with CSF
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        return
    if pointcloud is not None:
        print(">> Point cloud read successfully.\n")
        # Thinning
        thinned_pc = thin_pc(pointcloud, 10) # every 10th point
        print(">> Point cloud thinned.\n")
        # Outlier removal
        thinned_pc = knn_outlier_removal(thinned_pc, 10)  # k value for k-NN outlier removal
        print(">> Outliers removed.\n")
        # Ground filtering with CSF
        ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon)
        print (">> Ground points classified with CSF algorithm.\n")
            
        #test_ground_non_ground_separation(ground_points, non_ground_points)
        # Only show ground points
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='green', label='Ground Points', s=1)
        ax.set_title('Ground Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

        ## Step 2: Laplace Interpolation
        if ground_points.size == 0:
            print("No valid ground points found. Exiting...")
            return  
        
        dtm = laplace_interpolation(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
        print("DTM created and saved as dtm_laplace.tiff.")

        # Visualize the filtered DTM
        visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)
        print(">> Laplace interpolation complete.\n")

        # Save the ground points in a file called ground.laz
        save_ground_points_las(ground_points)
        print(">> Ground points saved to ground.laz.\n")
    
    print("\nStep 1 completed!\n\n")   

if __name__ == "__main__":
    main()
    