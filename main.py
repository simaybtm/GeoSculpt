'''
In this assignment you need to:

    1. automatically classify a 300mX300m region of the AHN4 point cloud into ground and non-ground using the CSF algorithm
    2. make a raster DTM using both kriging and Laplace interpolation
    3. compare your DTM rasters with the official AHN4 DTM raster
    4. to help with visualisation you’ll have to implement a contouring algorithm

You have to use the following Python libraries (others are not accepted, or ask me first): numpy, startinpy, pyinterpolate, rasterio, scipy, laspy, fiona, shapely.
'''

import step4
import step1
import step2
import step3


def run_all():
    # Ask user for the input file location
    dtm_input_file = input("Enter the path to the input file for DTM and step4: ")
    
    # Ask user for the output file location for GFLAP
    dtm_output_file = input("Enter the path to the output file for DTM: ")
    
    # Running step1
    print("Running Ground filtering + DTM creation with Laplace...")
    step1.main(dtm_input_file, dtm_output_file)
    print("Ground filtering + DTM creation with Laplace step is complete.")

    # Running step2
    print("Running Ground filtering + DTM creation with OK...")
    step2.main(dtm_input_file, dtm_output_file)
    
    # Running step3
    print("Running step3... comparing DTMs...")
    step3.main(dtm_output_file)

   # Running step4
    print("Running step4... extracting and saving contours...")
    step4.main(dtm_output_file)

    print("All steps are complete.")

if __name__ == "__main__":
    run_all()


























