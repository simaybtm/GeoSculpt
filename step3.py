import numpy as np
import rasterio
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Function to read raster and return masked array of elevation values
def read_elevation_data(dtm_path):
    with rasterio.open(dtm_path) as src:
        nodata = src.nodata
        elevation_data = src.read(1)
        if nodata is not None:
            elevation_data = np.ma.masked_equal(elevation_data, nodata)
        else:
            elevation_data = np.ma.masked_invalid(elevation_data)
            print("No nodata value found. Masking NaN values instead.")
    return elevation_data

def plot_3d_dtm(dtm_path):
    # Read the DTM data
    with rasterio.open(dtm_path) as src:
        elevation = src.read(1)  # Read the first band
        # Generate x and y coordinates for each cell
        x = np.arange(0, src.width)
        y = np.arange(0, src.height)
        x, y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the elevation data as the z-axis
    ax.plot_surface(x, y, elevation, cmap='terrain', edgecolor='none')
    
    ax.set_title('3D Visualization of DTM')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Elevation')
    
    plt.show()
    
# Paths to your DTMs
dtm_laplace_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\dtm_laplace.tiff"
dtm_ok_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\dtm_ordinary_kriging.tiff"
dtm_ahn_path = "C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\cropped_AHN.tif"

# Read the elevation data from each DTM
elevation_data_laplace = read_elevation_data(dtm_laplace_path)
elevation_data_ok = read_elevation_data(dtm_ok_path)
elevation_data_ahn = read_elevation_data(dtm_ahn_path)

# Calculate mean and standard deviation for each DTM
mean_laplace, std_laplace = np.mean(elevation_data_laplace), np.std(elevation_data_laplace)
mean_ok, std_ok = np.mean(elevation_data_ok), np.std(elevation_data_ok)
mean_ahn, std_ahn = np.mean(elevation_data_ahn), np.std(elevation_data_ahn)


# Print Total NaN values in each DTM
print("\n")
print(f"Total NaN values in Laplace DTM: {np.sum(elevation_data_laplace.mask)}")
print(f"Total NaN values in Ordinary Kriging DTM: {np.sum(elevation_data_ok.mask)}")
print(f"Total NaN values in Official AHN4 DTM: {np.sum(elevation_data_ahn.mask)}")


# Print the results
print("\n")
print(f"Laplace DTM - Mean: {mean_laplace}, Standard Deviation: {std_laplace}")
print(f"Ordinary Kriging DTM - Mean: {mean_ok}, Standard Deviation: {std_ok}")
print(f"Official AHN4 DTM - Mean: {mean_ahn}, Standard Deviation: {std_ahn}")
print("\n")

print("3D Visualization of Laplace DTM")
plot_3d_dtm(dtm_laplace_path)
print("3D Visualization of Ordinary Kriging DTM")
plot_3d_dtm(dtm_ok_path)
print("3D Visualization of Official AHN4 DTM")
plot_3d_dtm(dtm_ahn_path)
