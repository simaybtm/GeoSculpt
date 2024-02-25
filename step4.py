
#THIS IS step4.py

import numpy as np
import startinpy as st
import pyinterpolate as pi
import rasterio
import scipy
import laspy
import fiona
import shapely

'''
Implement the ideas described in “Conversion to isolines” in the terrainbook to produce a 
GeoPackage file of the contours lines at every meter (“full meters”, eg 1.0/2.0/3.0/etc).

Show me the results in the report and submit the code.

The Python code step4.py should also be submitted and have the following behaviour

name	step4.py
inputfile	GeoTIFF

and it should output a GeoPackage file called isocontours.gpkg where the lines have the 
elevation as an attribute.

You can create GeoPackage easily with “sister package” to rasterio called “fiona” (which uses the excellent “shapely”).


'''