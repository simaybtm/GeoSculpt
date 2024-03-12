
import numpy as np
import rasterio
import argparse
import matplotlib.pyplot as plt
import fiona
from fiona.crs import from_epsg
from shapely.geometry import mapping, LineString, box


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

def extract_and_save_contours(geo_tiff_path, output_gpkg_path, interval=1.0):
    with rasterio.open(geo_tiff_path) as src:
        elevation = src.read(1)
        affine = src.transform
        crs = src.crs
        raster_shape = elevation.shape

        if np.isnan(elevation).any():
            mean_elevation = np.nanmean(elevation[np.isfinite(elevation)])
            elevation[np.isnan(elevation)] = mean_elevation

        min_elevation = np.nanmin(elevation)
        max_elevation = np.nanmax(elevation)
        contour_levels = np.arange(min_elevation, max_elevation, interval)
        print(f"Extracting contours at levels: {contour_levels}")

        contours = plt.contour(elevation, levels=contour_levels, origin='upper', 
                               extent=[affine.c, affine.c + affine.a * raster_shape[1], 
                                       affine.f + affine.e * raster_shape[0], affine.f], 
                               linestyles='solid')
        print(f"Contours extracted from {geo_tiff_path}")
        
        # Check if none of the contours are touching each other
        for i, contour_path in enumerate(contours.collections):
            for path in contour_path.get_paths():
                line = LineString(path.vertices)
                if line.is_simple:
                    continue
                else:
                    print(f"Contour {i} is not simple")
                    break
     
        

        schema = {'geometry': 'LineString', 'properties': {'elevation': 'float'}}
        with fiona.open(output_gpkg_path, 'w', driver='GPKG', crs=crs, schema=schema) as dst:
            for i, contour_path in enumerate(contours.collections):
                for path in contour_path.get_paths():
                    line = LineString(path.vertices)
                    # Increase the length threshold for contours near the edge
                    length_threshold = 5 if is_near_edge(line, affine, raster_shape) else 1
                    if line.length > length_threshold:
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
    
                    
    