
#THIS IS step1.py

import numpy as np
import startinpy as st
import pyinterpolate as pi
import rasterio
import scipy
import laspy
import fiona
import shapely

import math

import argparse

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
epsilon	    threshold in meters to classify the ground

OUTPUTS:
dtm.tiff    representing the 50cm-resolution DTM of the area created with Laplace
ground.laz  the ground points
'''

### Step 1: Ground filtering + DTM creation with Laplace

## Read LAS file with laspy and filter points by bounds
def read_las(file_path, min_x, max_x, min_y, max_y):
    try:
        with laspy.open(file_path) as f:
            las = f.read()
            mask = (las.x >= min_x) & (las.x <= max_x) & (las.y >= min_y) & (las.y <= max_y)
            return np.column_stack((las.x[mask], las.y[mask], las.z[mask]))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to thin the point cloud by half
def thin_pc(pointcloud):
    return pointcloud[::10] 


# Function to compute geometric properties
def compute_geometric_properties(point, v1, v2, v3, triangle_normal):
    pass
    
    D = -np.dot(triangle_normal, v1)
    d_numerator = np.abs(np.dot(triangle_normal, point) + D)
    d_denominator = np.linalg.norm(triangle_normal)
    if d_denominator < 1e-6:
        return 0.0, 0.0
    d = d_numerator / d_denominator

    angles = []
    for p in [point, v1, v2, v3]:
        dot_product = np.dot(triangle_normal, p - v1)
        norms_product = np.linalg.norm(triangle_normal) * np.linalg.norm(p - v1)
        if norms_product < 1e-6:
            continue  # Avoid division by zero
        # Clamp the value within the range [-1, 1] to avoid NaNs
        angle = np.arccos(np.clip(dot_product / norms_product, -1, 1))
        angles.append(angle)

    return d, max(angles) if angles else 0.0


## Function to filter ground points using Laplace interpolation
#TODO



# Main function
def main(input_file_path, output_file_path):

    # Use parsed arguments with args.argument_name
    input_file_path = args.inputfile
    min_x = args.minx
    min_y = args.miny
    max_x = args.maxx
    max_y = args.maxy
    resolution = args.res
    csf_res = args.csf_res
    epsilon = args.epsilon

    # Print the input arguments
    print(f"Processing {input_file_path} with minx={min_x}, miny={min_y}, maxx={max_x}, maxy={max_y}, res={resolution}, csf_res={csf_res}, epsilon={epsilon}")

    # Processing pipeline
    pointcloud = read_las(input_file_path, min_x, max_x, min_y, max_y)
    if pointcloud is not None:
        thinned_pc = thin_pc(pointcloud)
        print("Point cloud read and thinned.")
        #ground_points = ground_filter_tin(thinned_pc, resolution, d_max, alpha_max)
        #save_ground_points_to_las(ground_points, output_file_path)

        print("Step 1 complete.")   