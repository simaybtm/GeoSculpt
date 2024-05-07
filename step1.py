
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

### --------------- Step 1: Ground filtering with CSF ---------------
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

## Function to thin the point cloud 
def thin_pc(pointcloud, thinning_percentage):
    print("Thinning the point cloud...")
    if thinning_percentage <= 0 or thinning_percentage >= 100:
        raise ValueError("Thinning percentage must be between 0 and 100 (exclusive).")

    # Calculate the number of points to keep based on the percentage
    num_points_to_keep = int(len(pointcloud) * (thinning_percentage / 100.0))
    
    # Ensure at least one point is kept
    num_points_to_keep = max(num_points_to_keep, 1)
    
    # Calculate the step size to achieve the desired thinning percentage
    step_size = len(pointcloud) // num_points_to_keep

    # Thinning by selecting points at regular intervals
    thinned_pointcloud = pointcloud[::step_size]  
    
    print(f" Thinning percentage: {thinning_percentage}%")
    print(f" Number of points before thinning: {pointcloud.shape[0]}")
    print(f" Number of points after thinning: {thinned_pointcloud.shape[0]}")

    ## PLOTS
    """
    # Plot the original point cloud in 3D (_0_)
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c='blue', s=1)
    ax.set_title('Original Point Cloud')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # Plot the thinned point cloud in 3D (_1_)
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

## Function to detect and remove outliers using k-Nearest Neighbors
def knn_outlier_removal(thinned_pointcloud, k):
    print("Detecting and removing outliers...")

    # Build KDTree for efficient neighbor search
    tree = cKDTree(thinned_pointcloud[:, :2])  # Use only X, Y for spatial queries

    # Compute the distances to the k-th nearest neighbor
    distances, _ = tree.query(thinned_pointcloud[:, :2], k=k + 1)  # k+1 because the point itself is included
    knn_distances = distances[:, k]  # We take the k-th nearest distance

    # Define threshold for outlier detection
    threshold = np.mean(knn_distances) + 2 * np.std(knn_distances)

    # Filter points where the k-th nearest neighbor is within the threshold
    non_outliers = thinned_pointcloud[knn_distances < threshold]

    print(f" Removed {len(thinned_pointcloud) - len(non_outliers)} outliers.")
    return non_outliers

## (USED in CSF) Function to run the Cloth Simulation Filter algorithm
class Vertex:
    def __init__(self, position, z_min, z_initial):
        self.position = position
        self.z = z_initial  # Start from an initial height 
        self.z_min = z_min
        self.previous_z = self.z
    # Update the z value of the vertex based on the displacement
    def update(self, displacement):
        new_z = self.z + displacement
        if new_z < self.z_min:
            new_z = self.z_min
        self.previous_z, self.z = self.z, new_z

class Edge:
    def __init__(self, vertex_a, vertex_b):
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b

    def relax(self):
        # A spring system where the edge tries to keep vertices at the average of their heights
        target_z = (self.vertex_a.z + self.vertex_b.z) / 2
        self.vertex_a.update((target_z - self.vertex_a.z) * 0.5)
        self.vertex_b.update((target_z - self.vertex_b.z) * 0.5)

## Function to run the Cloth Simulation Filter algorithm 
def cloth_simulation_filter(thinned_pointcloud, csf_res, epsilon, eps_z, max_iterations=10000):
    print("Running Cloth Simulation Filter algorithm...")

    if thinned_pointcloud.size == 0:
        print(" ERROR: Empty point cloud after filtering or thinning! Aborting CSF.")
        return np.array([]), np.array([])

    # Extract max and min values for grid initialization
    x_min, x_max = np.min(thinned_pointcloud[:, 0]), np.max(thinned_pointcloud[:, 0])
    y_min, y_max = np.min(thinned_pointcloud[:, 1]), np.max(thinned_pointcloud[:, 1])
    z_max = np.max(thinned_pointcloud[:, 2])

    # Initialize vertices
    x_range = np.arange(x_min, x_max, csf_res)
    y_range = np.arange(y_min, y_max, csf_res)
    vertices = [Vertex((x, y), z_max - 10, z_max + 10) for y in y_range for x in x_range] # Vertices are initialized at z_max + 10 and cannot go below z_max - 10,
    edges = []
    print(f" Grid initialized with {len(vertices)} vertices.")

    # Create edges between vertices in a grid
    width = len(x_range)
    height = len(y_range)
    for y in range(height):
        for x in range(width):
            if x > 0:  # Edge to the left
                edges.append(Edge(vertices[y * width + x - 1], vertices[y * width + x]))
            if y > 0:  # Edge above
                edges.append(Edge(vertices[(y - 1) * width + x], vertices[y * width + x]))

    kd_tree = cKDTree(thinned_pointcloud[:, :2])

    print(" Starting simulation loop with 'epsilon':", epsilon, "and 'eps_z':", eps_z, "\n")
    # Simulation loop to adjust the cloth 
    for iteration in tqdm(range(max_iterations), desc="Cloth Simulation Progress"):
        # Update vertices based on closest point cloud z values
        for vertex in vertices:
            _, index = kd_tree.query(vertex.position, k=1) # Find the closest point in the point cloud
            closest_point_z = thinned_pointcloud[index, 2] # Get the z value of the closest point
            vertex.update((closest_point_z - vertex.z) + epsilon) # Update the vertex z value

        # Relax edges 
        for edge in edges:
            edge.relax()

        # Check for convergence
        max_delta_z = max(abs(vertex.z - vertex.previous_z) for vertex in vertices)
        if max_delta_z < eps_z:
            print(f"    Convergence reached after {iteration + 1} iterations with max_delta_z: {max_delta_z}")
            break

    # Extract results
    ground_points = []
    non_ground_points = []
    for point in thinned_pointcloud:
        x_idx = np.searchsorted(x_range, point[0]) - 1
        y_idx = np.searchsorted(y_range, point[1]) - 1
        if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
            continue
        vertex = vertices[y_idx * width + x_idx]
        if abs(point[2] - vertex.z) <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)

    return np.array(ground_points), np.array(non_ground_points)

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

### --------------- Step 2: Laplace Interpolation ---------------
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
        subset_points = np.delete(ground_points, i, axis=0)  # Exclude the current point
        dtm = laplace_interpolation(subset_points, minx, maxx, miny, maxy, resolution)  # Re-run Laplace interpolation
        omitted_point = ground_points[i]
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

## (USED in Laplace) Function to check if a point is inside the convex hull
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

## -------------------- Main function --------------------
def main():
    # Use parsed arguments directly
    print("\n")
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, \
maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon} \n")
    
    eps_z = 0.0000005    # convergence threshold for stopping the simulation

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
    ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon, eps_z)
    print (">> Ground points classified with CSF algorithm.\n")
    # 4. Testing ground and non-ground points
    test_ground_non_ground_separation(ground_points, non_ground_points)
    #print(">> Testing ground and non-ground points complete.\n")

     #------ Step 2: Laplace Interpolation ------
    if ground_points.size == 0:
        print("No valid ground points found. Exiting program...")
        return  
    # 5. Laplace
    dtm = laplace_interpolation(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
    print("DTM created and saved as dtm_laplace.tiff.")

    # 6. Jackknife RMSE (computes Laplace again for each point and calculates RMSE)
    jackknife_error = jackknife_rmse_laplace(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
    print(f"Jackknife RMSE of Laplace interpolation: {jackknife_error}")
    print(">> Jackknife RMSE computed.\n")

    # 7. Visualize the filtered DTM
    visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)
    print("Shape of the DTM: ", dtm.shape)
    print(">> Laplace interpolation complete.\n")

    # 8. Save the ground points in a file called ground.laz
    save_ground_points_las(ground_points)
    print(">> Ground points saved to ground.laz.\n")
    
    print("\nStep 1 completed!\n\n")   

if __name__ == "__main__":
    main()
    