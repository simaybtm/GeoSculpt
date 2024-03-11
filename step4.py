
import numpy as np
import rasterio
import argparse
import matplotlib.pyplot as plt
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, LineString, shape


'''
INPUTS:
name	    step4.py
inputfile	GeoTIFF file representing the DTM (created with Laplace in step1.py or OK in step2.py)

OUTPUTS:
isocontours.gpkg    GeoPackage file of the contours lines at every meter
'''


# Argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Extract and save contours from a GeoTIFF file to a GeoPackage.')
parser.add_argument('inputfile', type=str, help='Input GeoTIFF file representing the DTM.')
args = parser.parse_args()

def extract_and_save_contours(geo_tiff_path, output_gpkg_path, interval=1.0):
    # Open the GeoTIFF file
    with rasterio.open(geo_tiff_path) as src:
        elevation = src.read(1)  # Read the first band into a 2D array
        affine = src.transform
        crs = src.crs  # Directly use the CRS of the input file
        
        # Ensure interval is positive and non-zero
        interval = max(interval, 1e-5)  # Prevent zero or negative interval

        print(elevation.min(), elevation.max())
        # Calculate min and max elevation values, ensuring they are finite
        min_elevation = np.nanmin(elevation)
        max_elevation = np.nanmax(elevation)
        if not np.isfinite(min_elevation) or not np.isfinite(max_elevation):
            print("Error: Elevation data contains non-finite values.")
            return
        
        # Adjust min and max elevations to ensure they span at least one interval
        if min_elevation == max_elevation:
            min_elevation -= 0.5 * interval
            max_elevation += 0.5 * interval
        
        # Generate contour lines at specified intervals
        contour_levels = np.arange(min_elevation, max_elevation, interval)
        if contour_levels.size == 0:
            print("Error: No contour levels could be generated.")
            return
        
        # Generate contours and process them
        contours = plt.contour(elevation, levels=contour_levels, origin='upper', 
                               extent=[affine.c, affine.c + affine.a * elevation.shape[1], 
                                       affine.f + affine.e * elevation.shape[0], affine.f], 
                               linestyles='solid')

        # Prepare GeoPackage output
        schema = {'geometry': 'LineString', 'properties': {'elevation': 'float'}}
        with fiona.open(output_gpkg_path, 'w', driver='GPKG', crs=crs, schema=schema) as dst:
            for i, contour_path in enumerate(contours.collections):
                for path in contour_path.get_paths():
                    line = LineString(path.vertices)
                    feature = {
                        'geometry': mapping(line),
                        'properties': {'elevation': contour_levels[i]},
                    }
                    dst.write(feature)
                    
    print(f"Contours saved to {output_gpkg_path}")
    return output_gpkg_path

## Main function
def main():
    # Use parsed arguments directly
    print(f"\nProcessing {args.inputfile} to create contours at every meter.\n")
    
    # Generate contours and save them to a GeoPackage
    extract_and_save_contours(args.inputfile, 'isocontours.gpkg')
    
    print("\nStep 4 complete!")
if __name__ == "__main__":
    main()
    
                    
    