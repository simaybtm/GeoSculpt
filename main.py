'''
In this assignment you need to:

    1. automatically classify a 300mX300m region of the AHN4 point cloud into ground and non-ground using the CSF algorithm
    2. make a raster DTM using both kriging and Laplace interpolation
    3. compare your DTM rasters with the official AHN4 DTM raster
    4. to help with visualisation you’ll have to implement a contouring algorithm

You have to use the following Python libraries (others are not accepted, or ask me first): numpy, startinpy, pyinterpolate, rasterio, scipy, laspy, fiona, shapely.
'''


'''
import numpy as np
import startinpy as st
import pyinterpolate as pi
import rasterio
import scipy
import laspy
import fiona
import shapely
'''

#THIS IS main.py


import step4
import step1



def run_all():
    # Ask user for the input file location
    #TODO: Change hardcoded input file while submitting
    #dtm_input_file = input("Enter the path to the input file for DTM and step4: ")
    dtm_input_file = r'C:\Users\simay\OneDrive\Desktop\DTM_R\DTM_creation\69EZ1_21.LAZ'
    
    # Ask user for the output file location for GFLAP
    #TODO: Change hardcoded input file while submitting
    #dtm_output_file = input("Enter the path to the output file for DTM: ")
    dtm_output_file = r'C:\Users\simay\OneDrive\Desktop\DTM_R\DTM_creation\dtm.tiff'
    
    # Running GFLAP main function
    print("Running Ground filtering + DTM creation with Laplace...")
    step1.main(dtm_input_file, dtm_output_file)
    print("Ground filtering + DTM creation with Laplace step is complete.")

    # Ask user for the output file location for step4
    step4_output_file = input("Enter the path to the output file for step4: ")
    
    # Running step4 main function
    print("Running Visualisation with iso-contours...")
    step4.main(dtm_input_file, step4_output_file)
    print("Visualisation with iso-contours complete.")

    '''
    # The output file from step4 is used as input for laplace_chm
    laplace_chm_input_file = step4_output_file

    # Running laplace_chm main function
    print("Running laplace_chm...")
    laplace_chm.main(laplace_chm_input_file, dtm_output_file)  # Passing TIFF from step4 and LAS from GFTIN
    print("laplace_chm processing complete.")
    '''

if __name__ == "__main__":
    run_all()


























