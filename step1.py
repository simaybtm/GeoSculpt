
# Test run with:  python step1.py 69GN2_14.LAZ 198350 308950 198600 309200 1 2 2

# You have to use the following Python libraries: numpy, startinpy, pyinterpolate, rasterio, scipy, laspy, fiona, shapely.
import numpy as np
import startinpy
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

import laspy

import argparse

# LIBRARIES FOR TESTING
import logging # testing output
import matplotlib.pyplot as plt # testing output
from mpl_toolkits.mplot3d import Axes3D # testing output
from tqdm import tqdm # Loading bar
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
thinned.laz the thinned + cropped according to the bbox input file
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

# Logging function  for debugging
def setup_logging():
    logging.basicConfig(filename='csf_simulation.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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

## Function to thin the point cloud using a grid-based approach
def thin_pc(pointcloud, keep_percentage):
    print("Thinning the point cloud...")
    
    # If percentage not between 0 and 100, error
    if keep_percentage < 0 or keep_percentage > 100:
        print(" ERROR: Invalid keep percentage. Must be between 0 and 100.")
        return np.array([])
    
    print(f" Thinning percentage (points kept): {keep_percentage}%")
    # Determine the dimensions of the point cloud's bounding box
    min_x, min_y = np.min(pointcloud[:, :2], axis=0)
    max_x, max_y = np.max(pointcloud[:, :2], axis=0)
    # Calculate the interval based on the desired percentage of points to keep
    interval = np.sqrt((max_x - min_x) * (max_y - min_y) / (len(pointcloud) * (keep_percentage / 100)))
    # Generate grid points
    x_coords = np.arange(min_x, max_x, interval)
    y_coords = np.arange(min_y, max_y, interval)
    grid_points = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])

    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(pointcloud[:, :2])
    # Find the nearest point in the point cloud to each grid point
    distances, indices = tree.query(grid_points, k=1)
    thinned_pointcloud = pointcloud[indices]
    thinned_pointcloud = np.unique(thinned_pointcloud, axis=0)  # Remove duplicates if necessary

    print(f" Number of points before thinning: {len(pointcloud)}")
    print(f" Number of points after thinning: {len(thinned_pointcloud)}")
    
    # Save thinned point cloud to a new LAS file  
    header = laspy.LasHeader(version="1.4", point_format=2)
    las = laspy.LasData(header)
    # Assign thinned points to the LAS
    las.x = thinned_pointcloud[:, 0]
    las.y = thinned_pointcloud[:, 1]
    las.z = thinned_pointcloud[:, 2]
    # Write the LAS file to disk
    las.write('thinned.laz')
    
    return thinned_pointcloud

## Function to detect and remove outliers using k-Nearest Neighbors
def knn_outlier_removal(thinned_pointcloud, k):
    print("Detecting and removing outliers...")

    # Build KDTree for efficient neighbor search
    tree = cKDTree(thinned_pointcloud[:, :2])  # Use only X, Y for spatial queries

    # Compute the distances to the k-th nearest neighbor
    distances, _ = tree.query(thinned_pointcloud[:, :2], k=k + 1)  # k+1 because the point itself is included
    knn_distances = distances[:, k] 
    
    # Define threshold for outlier detection
    threshold = np.mean(knn_distances) + 2 * np.std(knn_distances)

    # Filter points where the k-th nearest neighbor is within the threshold
    non_outliers = thinned_pointcloud[knn_distances < threshold]

    print(f" Removed {len(thinned_pointcloud) - len(non_outliers)} outliers.")
    return non_outliers

##  (USED in CSF) Function to create edges based on neighboring grid points
def neighbors(grid_x, grid_y):
    edge_list = []
    rows, cols = grid_x.shape
    # Iterate over the grid points and create edges
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col  # Convert 2D index to 1D index
            if col < cols - 1:  # Horizontal edge
                edge_list.append((idx, idx + 1))
            if row < rows - 1:  # Vertical edge
                edge_list.append((idx, idx + cols))
                
    return edge_list
        
## Function to run the Cloth Simulation Filter algorithm 
def cloth_simulation_filter(thinned_pointcloud, csf_res, epsilon, minx, maxx, miny, maxy):
    print("Running Cloth Simulation Filter algorithm...")

    if thinned_pointcloud.size == 0:
        print(" ERROR: Empty point cloud after filtering or thinning! Aborting CSF.")
        return np.array([]), np.array([])
    
    # Invert terrain
    max_z = np.max(thinned_pointcloud[:, 2])  
    inverted_pc = thinned_pointcloud.copy()
    inverted_pc[:, 2] = max_z - inverted_pc[:, 2]  # Invert the heights
    max_z_inverted = np.max(inverted_pc[:, 2])     # Max elevation in inverted cloth
    print(" Terrain inverted.")

    # Build a KDTree of the point cloud
    kd = cKDTree(inverted_pc[:, :2])  

    # Cloth grid points 
    print(" Creating the cloth.")
    x1 = np.arange (args.minx, args.maxx, csf_res)
    y1 = np.arange (args.miny, args.maxy, csf_res)
    grid_x, grid_y = np.meshgrid(x1, y1)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Create the cloth grid
    edge = np.array(neighbors(grid_x, grid_y))

    # Initialize vertices and edges of the cloth
    z1 = max_z_inverted + 10
    Cmin = inverted_pc[kd.query(grid_points, k=1)[1]][:, 2] # Minimum elevation of the nearest neighbor
    Ccurrent = np.linspace(z1, z1, grid_points.shape[0])
    Cprevious = np.full(grid_points.shape[0], z1)
    print(" Cloth initialized.")
    
    # Initiliaze movement of the cloth
    mobility = np.ones(grid_points.shape[0], dtype=bool) 
    update = np.inf
    
    # Start the simulation
    print(" Starting the simulation.")
    while update > 0.01:
            keep_going = mobility.nonzero()[0]
            Cprevious[keep_going] = Ccurrent[keep_going]
            Ccurrent[keep_going] -= 0.1  # gravity effect

            for e0, e1 in edge:
                if mobility[e0] or mobility[e1]: # If at least one of the vertices is mobile
                    ze0, ze1 = Ccurrent[e0], Ccurrent[e1]
                    average = (ze0 + ze1) / 2
                    if mobility[e0]: # If one of the vertices is mobile
                        Ccurrent[e0] = average
                    if mobility[e1]: # If the other vertex is mobile
                        Ccurrent[e1] = average
            
            not_moving = Ccurrent < Cmin
            Ccurrent[not_moving] = Cmin[not_moving]
            Cprevious[not_moving] = Cmin[not_moving]
            mobility = Ccurrent > Cmin
            
            update = np.max(np.abs(Ccurrent - Cprevious))
        
    print(" Cloth stabilized. Now starting classification of points as ground or non-ground.")
    
    # Adjust the cloth points to the original non-inverted elevations for accurate distance comparison
    cloth_points = np.column_stack((grid_points, max_z - Ccurrent))

    # Create KDTree from the cloth points for efficient nearest neighbor searches
    cloth_tree = cKDTree(cloth_points[:, :2])

    # Prepare arrays for ground and non-ground classification
    ground_points = []
    non_ground_points = []

    # Classify points based on their distance to the nearest cloth point
    for point in thinned_pointcloud:
        distance, index = cloth_tree.query([point[0], point[1]])  # Find nearest cloth point
        cloth_z = cloth_points[index, 2]
        vertical_distance = abs(point[2] - cloth_z)

        # Check against thresholds to classify points
        if vertical_distance <= epsilon:
            ground_points.append(point)
        else:
            non_ground_points.append(point)

    ground_points = np.array(ground_points)
    non_ground_points = np.array(non_ground_points)
    
    # Print Ground and Non-ground points with computed percentage to the total points
    print(f" Ground Points: {len(ground_points)} and this is \
        {len(ground_points) / len(thinned_pointcloud) * 100:.2f}% of the total points.")
    print(f" Non-Ground Points: {len(non_ground_points)} and this is\
        {len(non_ground_points) / len(thinned_pointcloud) * 100:.2f}% of the total points.")
    
    """
    # Plotting 2D Cloth vs Ground
    plt.figure(figsize=(15, 10))
    plt.scatter(cloth_points[:, 0], cloth_points[:, 1], c='red', s=1, label='Cloth Points')
    plt.scatter(ground_points[:, 0], ground_points[:, 1], c='green', s=1, label='Ground Points')
    plt.title('Cloth Simulation Filter (CSF) Classification')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
    # Plot 3D Ground and Non-Ground Points (_2_)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], c='green', label='Ground Points', s=1)
    ax.scatter(non_ground_points[:, 0], non_ground_points[:, 1], non_ground_points[:, 2], c='red', label='Non-Ground Points', s=1)
    ax.set_title('Ground and Non-Ground Points Classification')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    """
 
    return ground_points, non_ground_points

## (TESTING) Function to visualize the separation between ground and non-ground points
def test_ground_non_ground_separation(ground_points, non_ground_points):
    """
    Visualizes the separation between ground and non-ground points.
    
    Parameters:
    - ground_points: np.array, points classified as ground.
    - non_ground_points: np.array, points classified as non-ground.
    
    (We have to change numpy arrays to tuples because numpy arrays cannot be used directly in set operations.)
    """
    print("Starting testing of ground and non-ground points separation...")

   # Convert to set of tuples 
    ground_set = set(map(tuple, ground_points))
    non_ground_set = set(map(tuple, non_ground_points))
    
    # Find shared points
    shared_points = ground_set.intersection(non_ground_set)
    
    if shared_points:
        print(f" ERROR: Found {len(shared_points)} shared points between ground and non-ground.")
        shared_points = np.array(list(shared_points))  # Convert back to array for plotting
        
        if shared_points.size > 0:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(shared_points[:, 0], shared_points[:, 1], shared_points[:, 2], c='purple', label='Shared Points', s=1)
            ax.set_title('Shared Points between Ground and Non-Ground')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.show()
    else:
        print(" No shared points found between ground and non-ground! Test passed.\n")

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

## Function to remove stubborn outliers using TIN
def remove_outliers_with_tin(points):
    dt = startinpy.DT()
    for pt in points:
        dt.insert_one_pt(pt[0], pt[1], pt[2])

    # Find and remove triangles with long edges
    triangles = dt.triangles
    points = dt.points
    long_edge_threshold = 0.6
    to_remove = []

    for t in triangles:
        if not dt.is_finite(t): # Skip infinite triangles
            continue
        p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]
        if np.linalg.norm(p1 - p2) > long_edge_threshold or \
           np.linalg.norm(p2 - p3) > long_edge_threshold or \
           np.linalg.norm(p3 - p1) > long_edge_threshold:
            to_remove.extend([t[0], t[1], t[2]])

    # Removing points connected by long edges 
    for idx in set(to_remove): # Avoid duplicates
        dt.remove(idx) 
    
    # 'Collect garbage' to clean up the triangulation 
    if dt.has_garbage():
        dt.collect_garbage()

    # Return the cleaned-up points 
    cleaned_points = dt.points[1:]  # Exclude the infinite vertex
        
    print (f" Removed {len(points) - len(cleaned_points)} stubborn outliers.\n")
    
    # Plot cleaned points  
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cleaned_points[:, 0], cleaned_points[:, 1], cleaned_points[:, 2], c='blue', label='Cleaned Points', s=1)
    ax.set_title('Cleaned Points after Outlier Removal with TIN')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    """
    return cleaned_points

### --------------- Step 2: Laplace Interpolation ---------------
## Function to compute the Jackknife RMSE for Laplace interpolation
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
    """
    Create the continuous 0.50cm-resolution DTM of this 250mX250m area using Laplace interpolation.

    Input:
        ground_points: np.array, array of ground points with columns [x, y, z].
        minx, maxx, miny, maxy: float, bounding box coordinates.
        resolution: float, resolution of the DTM grid.
    Output:
        dtm: np.array, a 2D array representing the DTM.

    IMPORTANT: You are not allowed to use startinpy’s interpolate() function, you need to implement 
               it yourself (but you can use startinpy triangulation and other functions).
    """
    dt = startinpy.DT()
    for pt in ground_points:
        dt.insert_one_pt(pt[0], pt[1], pt[2])
    # Preparing grid for interpolation
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    dtm = np.full(grid_x.shape, np.nan)
    # Perform Delaunay triangulation using startinpy
    print(" Performing Delaunay triangulation...")
    triangulation = startinpy.DT()
    for pt in ground_points:
        triangulation.insert_one_pt(pt[0], pt[1], pt[2])

    # Build a k-d tree for fast point-in-hull check
    hull_points = ground_points[:, :2]
    hull = ConvexHull(hull_points)
    kdtree = cKDTree(hull_points)

    # Adding progress bar
    total_points = grid_x.shape[0] * grid_x.shape[1]
    progress_bar = tqdm(total=total_points, desc="Interpolating DTM")

    insufficient_neighbors_count = 0
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            point = [grid_x[i, j], grid_y[i, j]]
            if not point_in_hull(point, hull):
                continue  # Skip points outside the convex hull

            # Use k-d tree to find the closest point and its neighbors
            dist, idx = kdtree.query(point, k=1)
            closest_idx = idx
            neighbors = triangulation.adjacent_vertices_to_vertex(closest_idx)
            neighbors = neighbors[neighbors != 0]  # Exclude the infinite vertex

            if len(neighbors) < 3:  # If not enough neighbors, skip this point
                insufficient_neighbors_count += 1
                continue

            neighbor_points = triangulation.points[neighbors]
            if len(neighbor_points) == 0:
                continue

            # Compute Voronoi weights for the neighbors of the current point
            weights = np.zeros(len(neighbor_points))
            total_weight = 0
            for k in range(len(neighbor_points)):
                voronoi_edge_length = compute_voronoi_edge_length(triangulation, neighbors[k])
                delaunay_edge_length = point_dist(neighbor_points[k, :2], point)
                if delaunay_edge_length != 0:
                    weights[k] = voronoi_edge_length / delaunay_edge_length
                    total_weight += weights[k]
            if total_weight > 0:
                weights /= total_weight
            else:
                continue

            # Perform interpolation using the weights
            z_value = np.dot(weights, neighbor_points[:, 2])
            dtm[i, j] = z_value

            progress_bar.update(1)  # Update the progress bar

    progress_bar.close()  # Close the progress bar when done
    print(f"Number of NaNs in the DTM: {np.isnan(dtm).sum()}")
    print(f"Number of points with insufficient neighbors: {insufficient_neighbors_count}")

    # Save the DTM to a TIFF file
    transform = from_origin(minx, maxy, resolution, -resolution)  
    with rasterio.open('dtm.tiff', 'w', driver='GTiff',
                       height=dtm.shape[0], width=dtm.shape[1],
                       count=1, dtype=str(dtm.dtype), crs='EPSG:28992', # Dutch National Grid (Amersfoort / RD New)
                       transform=transform) as dst:
        dst.write(dtm, 1)
    
    return dtm

## (USED in Laplace) Check if a point is inside the convex hull
def point_in_hull(point, hull):
    return all((np.dot(eq[:-1], point) + eq[-1]) <= 1e-12 for eq in hull.equations)

## (USED in Laplace) Calculates the circumcircle center for a given triangle, which is used to determine Voronoi edge lengths
def circum_circle(triangulation, triangle):
    pts = triangulation.points[triangle]
    A = pts[0]
    B = pts[1]
    C = pts[2]
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    # Calculate circumcenter coordinates
    Ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) + (B[0]**2 + B[1]**2) * (C[1] - A[1]) + (C[0]**2 + C[1]**2) * (A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) + (B[0]**2 + B[1]**2) * (A[0] - C[0]) + (C[0]**2 + C[1]**2) * (B[0] - A[0])) / D
    return np.array([Ux, Uy])

## (USED in Laplace) Calculate the Euclidean distance between two points
def point_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

## (USED in Laplace) Compute the length of the Voronoi edge for a given point
def compute_voronoi_edge_length(triangulation, vertex_index):
    neighbors = triangulation.adjacent_vertices_to_vertex(vertex_index)
    total_length = 0
    for i in range(len(neighbors) - 1):
        edge_length = point_dist(triangulation.points[neighbors[i], :2], triangulation.points[neighbors[i+1], :2])
        total_length += edge_length
    return total_length / len(neighbors)


def startinpy_laplace(ground_points, minx, maxx, miny, maxy, resolution):
    dt = startinpy.DT()
    for pt in ground_points:
        dt.insert_one_pt(pt[0], pt[1], pt[2])
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    locations = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    interpolated_values = dt.interpolate({"method": "Laplace"}, locations, strict=False)
    dtm_startinpy = interpolated_values.reshape(grid_x.shape)
     # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, dtm_startinpy, cmap='terrain', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title('Digital Terrain Model (DTM) Interpolated with startinpy Laplace')
    plt.show()
    return dtm_startinpy

## -------------------- Main function --------------------
def main():
    # Use parsed arguments directly
    print("\n")
    print(f"Processing {args.inputfile} with minx={args.minx}, miny={args.miny}, maxx={args.maxx}, \
maxy={args.maxy}, res={args.res}, csf_res={args.csf_res}, epsilon={args.epsilon} \n")
    
    setup_logging()

    #------ Step 1: Ground filtering with CSF ------
    pointcloud = read_las(args.inputfile, args.minx, args.maxx, args.miny, args.maxy)
    if pointcloud is None or pointcloud.size == 0:
        print("No points found within the specified bounding box.")
        # Terminating the program if no points are found
        return
    else:
        print(">> Point cloud read successfully.\n")

    # 1. Thinning
    thinned_pc = thin_pc(pointcloud, 50) # Percentage of points to keep
    print(">> Point cloud thinned.\n")
    # 2. Outlier removal
    thinned_pc = knn_outlier_removal(thinned_pc, 10)  # k value for k-NN outlier removal
    print(">> Outliers removed succesfully.\n")
    # 3. Ground filtering with CSF
    ground_points, non_ground_points = cloth_simulation_filter(thinned_pc, args.csf_res, args.epsilon, \
        args.minx, args.maxx, args.miny, args.maxy)
    print ("\n>> Ground points classified with CSF algorithm.\n")
    # ADDITIONAL: Testing ground and non-ground points
    #test_ground_non_ground_separation(ground_points, non_ground_points)
    #print(">> Testing ground and non-ground points complete.\n")
    
    # ADDITIONAL: Remove stubborn outliers with TIN
    print("Removing stubborn outliers with TIN...")
    ground_points = remove_outliers_with_tin(ground_points)

     #------ Step 2: Laplace Interpolation ------
    if ground_points.size == 0:
        print("No valid ground points found. Exiting program...")
        return    
    
    
    # Startinpy Laplace interpolation
    #print("Starting Laplace interpolation with startinpy...")
    #dtm_startinpy = startinpy_laplace(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
    #print(">> Laplace interpolation with startinpy complete.\n")

    # 4. Laplace
    print("Starting Laplace interpolation...")
    dtm = laplace_interpolation(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
    print("DTM created and saved as dtm_laplace.tiff.")

    # ADDITIONAL: Jackknife RMSE (computes Laplace again for each point and calculates RMSE)
    #jackknife_error = jackknife_rmse_laplace(ground_points, args.minx, args.maxx, args.miny, args.maxy, args.res)
    #print(f"Jackknife RMSE of Laplace interpolation: {jackknife_error}")
    #print(">> Jackknife RMSE computed.\n")

    # ADDITIONAL: Visualize the filtered DTM
    visualize_laplace(dtm, args.minx, args.maxx, args.miny, args.maxy, args.res)
    print(" Shape of the DTM: ", dtm.shape)
    print(">> Laplace interpolation complete.\n")

    # 5. Save the ground points in a file called ground.laz
    save_ground_points_las(ground_points)
    print(">> Ground points saved to ground.laz.\n")
    
    print("\nStep 1 completed!\n\n")   
    
if __name__ == "__main__":
    main()
