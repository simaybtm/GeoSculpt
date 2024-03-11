# THIS IS step3.py

'''
Compare your two created DTMs
1. with each other (Laplace versus OK),
2. and with the official AHN4 file (0.5m DTM).

# Highlight their differences and try to explain why they are different in the report.
'''
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Compare DTMs.')
parser.add_argument('laplace_dtm', type=str, help='Input Laplace DTM GeoTIFF file')
parser.add_argument('ok_dtm', type=str, help='Input OK DTM GeoTIFF file')
parser.add_argument('ahn4', type=str, help='Input AHN4 GeoTIFF file')  # Assuming AHN4 is also in GeoTIFF format
args = parser.parse_args()

def read_dtm(file_path):
    with rasterio.open(file_path) as src:
        dtm = src.read(1)
        # Replace no-data values with NaN. Adjust the no-data value as necessary.
        no_data_value = src.nodata
        if no_data_value is not None:
            dtm[dtm == no_data_value] = np.nan
    return dtm

def crop_ahn4(file_path, laplace_shape, ok_shape):
    with rasterio.open(file_path) as src:
        width, height = src.width // 2, src.height // 2
        window = rasterio.windows.Window(width // 2, height, width // 2, height // 2)
        cropped_image = src.read(1, window=window)
    return cropped_image

def compute_statistics(dtm):
    """Compute basic statistics of a DTM, ignoring NaN values."""
    stats = {
        'mean': np.nanmean(dtm),
        'median': np.nanmedian(dtm),
        'std': np.nanstd(dtm),
        'min': np.nanmin(dtm),
        'max': np.nanmax(dtm[np.isfinite(dtm)]),  # Ensure max is computed on finite values
    }
    return stats

def print_statistics(stats, title):
    print(f"Statistics for {title}:")
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}")
    print("\n")

def plot_histograms_side_by_side(dtm1, dtm2, dtm3, titles):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].hist(dtm1.flatten(), bins=50, color='blue', alpha=0.7)
    ax[0].set_title(f"Histogram - {titles[0]}")
    ax[1].hist(dtm2.flatten(), bins=50, color='green', alpha=0.7)
    ax[1].set_title(f"Histogram - {titles[1]}")
    ax[2].hist(dtm3.flatten(), bins=50, color='red', alpha=0.7)
    ax[2].set_title(f"Histogram - {titles[2]}")
    for a in ax:
        a.set_xlabel('Elevation (m)')
        a.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def main():
    laplace_dtm = read_dtm(args.laplace_dtm)
    ok_dtm = read_dtm(args.ok_dtm)
    ahn4_cropped = crop_ahn4(args.ahn4, laplace_dtm.shape, ok_dtm.shape)

    laplace_stats = compute_statistics(laplace_dtm)
    ok_stats = compute_statistics(ok_dtm)
    ahn4_stats = compute_statistics(ahn4_cropped)

    print_statistics(laplace_stats, 'Laplace DTM')
    print_statistics(ok_stats, 'OK DTM')
    print_statistics(ahn4_stats, 'Cropped AHN4 DTM')

    plot_histograms_side_by_side(laplace_dtm, ok_dtm, ahn4_cropped, ['Laplace DTM', 'OK DTM', 'Cropped AHN4 DTM'])

if __name__ == '__main__':
    main()