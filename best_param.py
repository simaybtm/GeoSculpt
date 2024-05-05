import subprocess
import numpy as np

# Define the range and increments for the parameters
resolution_range = np.arange(1.0, 5.1, 1.0)  # Example: from 1.0m to 5.0m resolution in 1m steps
csf_res_range = np.arange(1.0, 5.1, 1.0)     # Same ranges for CSF resolution
epsilon_range = np.arange(0.1, 1.1, 0.2)     # Epsilon from 0.1 to 1.0 in 0.2 steps

# Define the bounds of the area (fixed in this example)
minx, miny = 198350, 308950
maxx, maxy = 198600, 309200
inputfile = "69GN2_14.LAZ"

# Path to your Python environment and script
python_path = "python"
script_path = "step1.py"

# Function to run the script
def run_script(res, csf_res, epsilon):
    cmd = [
        python_path, script_path, inputfile,
        str(minx), str(miny), str(maxx), str(maxy),
        str(res), str(csf_res), str(epsilon)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return result.stdout

# Loop over all combinations of parameters
for res in resolution_range:
    for csf_res in csf_res_range:
        for epsilon in epsilon_range:
            print(f"Running with res={res}, csf_res={csf_res}, epsilon={epsilon}")
            output = run_script(res, csf_res, epsilon)
            print(output)

            # Here you might want to parse 'output' to look for metrics of interest,
            # or modify `run_script` to save output metrics to a file and analyze them later.
            # For example, you could save the RMSE of the DTM and the number of ground points
            # to a CSV file for further analysis.

# Once the loop finishes, you can analyze the results to find the best parameters
# For example, you could look for the lowest RMSE or the highest number of ground points
# and print the corresponding parameters.

# Example: Find the best parameters based on the RMSE
best_rmse = float('inf')
best_params = None

for res in resolution_range:
    for csf_res in csf_res_range:
        for epsilon in epsilon_range:
            # Parse the output to extract the RMSE
            # For example, if the output is in the format "RMSE: 0.1234"
            rmse = float(output.split(':')[-1])
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (res, csf_res, epsilon)

print(f"Best parameters: {best_params} with RMSE: {best_rmse}")





