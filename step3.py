import numpy as np
import rasterio
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import laspy
# Loading bar
from tqdm import tqdm

# Paths to your DTMs
dtm_laplace_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\dtm_laplace.tiff"
dtm_ok_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\dtm_ordinary_kriging.tiff"
dtm_ahn_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\cropped_AHN.tif"

# Function to read raster and return masked array of elevation values
def read_elevation_data(dtm_path):
    with rasterio.open(dtm_path) as src:
        nodata = src.nodata
        elevation_data = src.read(1)
        if nodata is not None:
            elevation_data = np.ma.masked_equal(elevation_data, nodata)
        else:
            elevation_data = np.ma.masked_invalid(elevation_data)
    return elevation_data

# Print shappe of each TIFF file
print("\n")
print(f"Shape of Laplace DTM: {read_elevation_data(dtm_laplace_path).shape}")
print(f"Shape of Ordinary Kriging DTM: {read_elevation_data(dtm_ok_path).shape}")
print(f"Shape of Official AHN4 DTM: {read_elevation_data(dtm_ahn_path).shape}")
print("\n")


# Calculate mean and standard deviation for each DTM
mean_laplace, std_laplace = np.mean(read_elevation_data(dtm_laplace_path)), np.std(read_elevation_data(dtm_laplace_path))
mean_ok, std_ok = np.mean(read_elevation_data(dtm_ok_path)), np.std(read_elevation_data(dtm_ok_path))
mean_ahn, std_ahn = np.mean(read_elevation_data(dtm_ahn_path)), np.std(read_elevation_data(dtm_ahn_path))

def read_laz_file(filepath):
    """ Reads a .laz file and returns the x, y, z coordinates as a numpy array. """
    with laspy.open(filepath) as file:
        las = file.read()
        points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

# Path to your .laz file
filepath = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\ground.laz"
points = read_laz_file(filepath)
print("\n\nPoint cloud data of the terrain loaded, number of points:", points.shape[0])

def inverse_distance_weighting(x, y, z, xi, yi, power=2):
    """ Interpolate the z value at coordinates (xi, yi) using IDW from surrounding points (x, y, z). """
    distances = np.sqrt((x - xi)**2 + (y - yi)**2)
    if np.any(distances == 0):
        return z[distances == 0][0]
    weights = 1 / distances**power
    return np.sum(weights * z) / np.sum(weights)

def jackknife_rmse(points, power=2):
    errors = []
    for i in tqdm(range(len(points)), desc="Computing Jackknife RMSE"):
        test_point = points[i]
        train_points = np.delete(points, i, axis=0)
        x_train, y_train, z_train = train_points[:, 0], train_points[:, 1], train_points[:, 2]
        z_pred = inverse_distance_weighting(x_train, y_train, z_train, test_point[0], test_point[1], power=power)
        error = (z_pred - test_point[2]) ** 2
        errors.append(error)
    rmse = np.sqrt(np.mean(errors))
    return rmse

# Proceed with the jackknife RMSE computation
#rmse = jackknife_rmse(points)
#print("\nInitiating Jackknife resampling for RMSE computation...")
#print(f"Jackknife RMSE: {rmse}")

# Print Total NaN values in each DTM
print("\n")
print(f"Total NaN values in Laplace DTM: {np.sum(read_elevation_data(dtm_laplace_path).mask)}")
print(f"Total NaN values in Ordinary Kriging DTM: {np.sum(read_elevation_data(dtm_ok_path).mask)}")
print(f"Total NaN values in Official AHN4 DTM: {np.sum(read_elevation_data(dtm_ahn_path).mask)}")


# Print the results
print("\n")
print(f"Laplace DTM - Mean: {mean_laplace}, Standard Deviation: {std_laplace}")
print(f"Ordinary Kriging DTM - Mean: {mean_ok}, Standard Deviation: {std_ok}")
print(f"Official AHN4 DTM - Mean: {mean_ahn}, Standard Deviation: {std_ahn}")
print("\n")

def calculate_rmse(true_values, predicted_values):
    # Ensure both arrays have the same mask applied
    valid_mask = ~np.ma.getmaskarray(true_values) & ~np.ma.getmaskarray(predicted_values)
    true_values_valid = true_values[valid_mask]
    predicted_values_valid = predicted_values[valid_mask]

    # Calculate RMSE only on valid, overlapping data
    return np.sqrt(mean_squared_error(true_values_valid, predicted_values_valid))


# rmse_laplace = calculate_rmse(elevation_data_ahn, elevation_data_laplace)
# rmse_ok = calculate_rmse(elevation_data_ahn, elevation_data_ok)

# print(f"RMSE between Official AHN4 DTM and Laplace DTM: {rmse_laplace}")
# print(f"RMSE between Official AHN4 DTM and Ordinary Kriging DTM: {rmse_ok}")

