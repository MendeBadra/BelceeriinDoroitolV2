import cv2
import numpy as np
import rasterio
import pandas as pd

from tqdm import tqdm
import importlib
import sys
from pathlib import Path
from typing import Union

import align_dji_images
importlib.reload(align_dji_images)
from align_dji_images import convert_to_8bit, align_with_ecc, align_with_warp_matrix, calculate_msavi, calculate_ndvi, normalize_16bit, nan_outside_range

def save_multiband_geotiff(data_bands, band_names, output_path, filename_prefix="DJI", drone_image_number="000", name="BANDS"):
    """
    Ene funks n 5 band aa avaad, -> DJI_***BANDS.TIF 1300x1600x5
    Msavi, ndvi -> DJI_***INDS.TIF 
    Sul tal: Arai uuruur. 

    Save a multi-band NumPy array to a GeoTIFF with band descriptions.

    Parameters:
        data_bands (np.ndarray): Array of shape (bands, height, width)
        band_names (list of str): List of band names, one per band
        output_path (str or Path): Directory to save the TIFF
        filename_prefix (str): Optional prefix for the filename
        drone_image_number (str or int): Identifier for the image (used in filename)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"{filename_prefix}_{drone_image_number}{name}.TIF"

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data_bands.shape[1],
        width=data_bands.shape[2],
        count=data_bands.shape[0],
        dtype=data_bands.dtype,
        compress="LZW"
    ) as dst:
        for i in range(data_bands.shape[0]):
            dst.write(data_bands[i], i + 1)
            dst.set_band_description(i + 1, band_names[i])

def process_images(input_dir: Union[str, Path], output_dir: Union[str, Path], drone_image_number: str):
    image_path = Path(input_dir)
    image_names = (image_path).glob(f"DJI_{drone_image_number}?.TIF")
    image_names = sorted(list(image_names)) # Big mistake to think that pattern matching would sort the files in order
    band_names = ['Blue', 'Green', 'Red', 'Red Edge', 'NIR'] # in order
    print("Found following images:")

    for index, image in enumerate(image_names):
        print(f"{band_names[index]} [{index}]: {image}")
        # Open and read images
    images_raw = [cv2.imread(str(image), cv2.IMREAD_UNCHANGED) for image in image_names]
    images_16bit = [normalize_16bit(image) for image in images_raw]
    images = [convert_to_8bit(image) for image in images_16bit]
    blue = images[0]
    green = images[1]
    red = images[2]
    #image1 = convert_to_8bit(images[4])
    red_edge = images[3]
    nir = images[4]


    print("Using red as reference image: ", image_names[2])
    blue_aligned, blue_warp = align_with_ecc(red, blue)
    green_aligned, green_warp = align_with_ecc(red, green)
    red_edge_aligned, red_warp = align_with_ecc(red, red_edge)
    nir_aligned, nir_warp = align_with_ecc(red, nir)
    # Align reflectance bands according to the warp matrix of each band
    blue = images_raw[0]
    green = images_raw[1]
    red = images_raw[2]
    #image1 = convert_to_8bit(images[4])
    red_edge = images_raw[3]
    nir = images_raw[4]

    blue_aligned = align_with_warp_matrix(red, blue, blue_warp)
    green_aligned = align_with_warp_matrix(red, green, green_warp)
    red_edge_aligned = align_with_warp_matrix(red, red_edge, red_warp)
    nir_aligned = align_with_warp_matrix(red, nir, nir_warp)
    ndvi, ndvi_scaled = calculate_ndvi(nir_aligned, red)
    msavi, msavi_scaled = calculate_msavi(nir_aligned, red)

    msavi = nan_outside_range(msavi, -1, 1)
    ndvi = nan_outside_range(ndvi, -1, 1)
    data_bands = np.stack([blue_aligned, green_aligned, red, red_edge_aligned, nir_aligned], axis=-1)
    data_bands = np.permute_dims(data_bands, (2, 0, 1))   # needed for rasterio to save
    data_indices = np.stack([msavi, ndvi], axis=-1)
    data_indices = np.permute_dims(data_indices, (2, 0, 1))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Saving aligned bands to {output_path} directory...")
    save_multiband_geotiff(data_bands,
                           band_names=band_names, 
                           output_path=output_path, 
                           filename_prefix="DJI", 
                           drone_image_number=drone_image_number, 
                           name="BANDS")
    save_multiband_geotiff(data_indices, 
                           band_names=["MSAVI", "NDVI"], 
                           output_path=output_path, 
                           filename_prefix="DJI", 
                           drone_image_number=drone_image_number, 
                           name="INDS")


if __name__ == "__main__":
    # Directory paths
    # root_dir = Path("/media/razydave/CenAppMath/MendeCenMathApp/drone_data_2024_october_reflectance")
    # aligned_root_dir = Path("drone_data_2024_october_aligned")

    if len(sys.argv) != 3:
        print("Usage: python align_images_batch.py <input_dir> <output_dir> <drone_image_number>")
        exit(1)
    
    root_dir = Path(sys.argv[1])
    aligned_root_dir = Path(sys.argv[2])

    # Iterate over all directories (Best_10, Bad_10, etc.)
    for category_dir in tqdm(root_dir.iterdir(), desc="Processing categories"):
        if category_dir.is_dir():
            # Iterate over each subfolder (e.g., 105FPLAN, 106FPLAN, etc.)
            for dji_dir in tqdm(category_dir.iterdir(), desc=f"Processing DJI sets in {category_dir}", leave=False):
                if dji_dir.name == "DJI_P4_MS":
                    for fplan_dir in tqdm(dji_dir.iterdir(), desc=f"Processing FPLANS in {dji_dir} of {category_dir}", leave=False):
                        if fplan_dir.is_dir():
                            # Collect the TIF files related to the current FPLAN
                            tif_files = sorted(fplan_dir.glob("DJI_*.TIF"))
                            
                            # Group images in batches of 5
                            for i in tqdm(range(0, len(tif_files), 5), desc=f"Processing batches in {fplan_dir.name}", leave=False):
                                # Get the batch of 5 images
                                batch = tif_files[i:i+5]
                                
                                # Assuming that the drone_image_number is the first three digits of the file name (e.g., '027' from 'DJI_0275.TIF')
                                drone_image_number = batch[0].name[4:7]  # Extracts '027' from 'DJI_0275.TIF'
                                
                                # Input and output directories based on your structure
                                input_dir = fplan_dir
                                output_dir = aligned_root_dir / category_dir.name / fplan_dir.name
                                
                                # Call the process_images function
                                process_images(input_dir, output_dir, drone_image_number)