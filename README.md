# DTM_creation

## GEO1015.2023 - Point Clouds, DTMs, and Visualisation

This repository contains a Python project developed as part of the GEO1015 course, focusing on the processing and analysis of point clouds to generate Digital Terrain Models (DTMs) and their visualization. The project emphasizes the automatic classification of ground and non-ground points within a specified area, the creation of DTMs using different interpolation techniques, and the comparison and visualization of the generated DTMs.

## Overview

The project is structured around several key steps:

1. **Ground Filtering and DTM Creation with Laplace Interpolation**: Implements the Cloth Simulation Filter (CSF) algorithm to classify ground points and generates a DTM using Laplace interpolation.
2. **DTM Creation with Ordinary Kriging (OK)**: Uses the ground points classified in step 1 to create a DTM using Ordinary Kriging interpolation.
3. **Comparison of DTMs**: Compares the DTMs generated through Laplace interpolation and Ordinary Kriging with each other and with an official AHN4 DTM raster.
4. **Visualization with Iso-Contours**: Implements an algorithm to generate iso-contours from the DTM for visualization purposes.

### The Dataset

The project uses a 300m x 300m area within the Netherlands, specified by the following bounding box:

- Min: (190250.0, 313225.0)
- Max: (190550.0, 313525.0)

This area is located within tile 69EZ1_21 of the GeoTiles dataset.

## Requirements

The project relies on several Python libraries for its implementation:

- Numpy
- StartinPy
- Pyinterpolate
- Rasterio
- Scipy
- Laspy
- Fiona
- Shapely

## Usage

The repository includes several Python scripts, each corresponding to a step in the project:

- `step1.py`: Ground filtering and DTM creation with Laplace interpolation. DTM creation with Ordinary Kriging.
- `step4.py`: Visualization script for generating iso-contours.
- `main.py`: Running the code.

To run a script, use the following command structure (example for `step1.py`):

```shell
python step1.py <inputfile.laz> <minx> <miny> <maxx> <maxy> <resolution> <csf_res> <epsilon>
