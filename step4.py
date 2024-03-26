
import numpy as np
import rasterio
import argparse
import matplotlib.pyplot as plt
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, LineString, box
from shapely.ops import unary_union

from scipy.ndimage import gaussian_filter



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

def is_near_edge(line, affine, shape, buffer_distance=15):
    """
    Checks if a line is within a certain buffer distance from the edges of the raster.
    """
    raster_box = box(affine.c, affine.f, affine.c + affine.a * shape[1], affine.f + affine.e * shape[0])
    edge_buffer = raster_box.buffer(-buffer_distance) # Contract the raster boundary by buffer_distance
    return not line.within(edge_buffer)

def extract_and_save_contours(geo_tiff_path, output_gpkg_path, interval=1.0, simplify_tolerance=0.5, buffer_distance=15, sigma=1):
    with rasterio.open(geo_tiff_path) as src:
        elevation = src.read(1, masked=True) # Read the first band (index 0) and mask no data values
        affine = src.transform
        crs = src.crs

        # Apply Gaussian smoothing
        # elevation = gaussian_filter(elevation.filled(fill_value=np.nanmean(elevation)), sigma=1)

        min_elevation, max_elevation = np.nanpercentile(elevation, [2, 98])  # This will exclude the top and bottom 2% of values (outliers probably)
        contour_levels = np.arange(min_elevation, max_elevation + interval, interval)  # Ensure max elevation is included in the range
        
        if contour_levels.size == 0:
            print("Error: No contour levels could be generated.")
            return

        print(f"Attempting to generate contours for levels: {contour_levels}")

        contours = plt.contour(elevation, levels=contour_levels, origin='upper',
                               extent=[affine.c, affine.c + affine.a * elevation.shape[1],
                                       affine.f + affine.e * elevation.shape[0], affine.f],
                               linestyles='solid')

        schema = {'geometry': 'LineString', 'properties': {'elevation': 'float'}}
        with fiona.open(output_gpkg_path, 'w', driver='GPKG', crs=crs, schema=schema) as dst:
            for i, contour in enumerate(contours.allsegs):
                for seg in contour:
                    if seg.size > 0: # Skip empty segments
                        line = LineString(seg)
                        feature = {
                            'geometry': mapping(line),
                            'properties': {'elevation': contour_levels[i]}
                        }
                        dst.write(feature)

        print(f"Contours saved to {output_gpkg_path}")

## Main function
def main():
    # Use parsed arguments directly
    print(f"\nProcessing {args.inputfile} to create contours lines.\n")
    
    # Generate contours and save them to a GeoPackage
    extract_and_save_contours(args.inputfile, 'isocontours.gpkg')
    
    print("\nStep 4 complete!")
if __name__ == "__main__":
    main()
    
                    
    