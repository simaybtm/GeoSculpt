import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd

def crop_raster(input_raster, output_raster, minx, miny, maxx, maxy):
    """
    Crops a raster file to the specified bounding box.
    
    Parameters:
    - input_raster: Path to the input raster file.
    - output_raster: Path where the cropped raster will be saved.
    - minx, miny, maxx, maxy: Coordinates of the bounding box.
    """
    
    # Create a bounding box from the provided coordinates
    bbox = box(minx, miny, maxx, maxy)
    
    # Convert the bounding box to a GeoDataFrame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:28992")
    
    with rasterio.open(input_raster) as src:
        # Crop the image
        out_image, out_transform = mask(src, geo.geometry, crop=True)
        out_meta = src.meta.copy()
        
        # Update the metadata to reflect the new shape (width, height), transform, and CRS
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        # Write the cropped image to a new file
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(out_image)

# Define the coordinates of your bounding box
minx, miny, maxx, maxy = 190250.0, 313225.0, 190550.0, 313525.0

# Specify the path to your input and output raster files
input_raster = 'C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\AHN4_69EZ1.tif'
output_raster = 'C:\\Users\\simay\\OneDrive\\Desktop\\DTM_R\\DTM_creation\\cropped_AHN.tif'

# Crop the raster
crop_raster(input_raster, output_raster, minx, miny, maxx, maxy)
print("Raster cropped successfully!")
