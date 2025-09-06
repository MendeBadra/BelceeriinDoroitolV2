# use num_env for this code!!!
import os, gc
# import numpy as np
import cupy as cp
import pandas as pd

from matplotlib import pyplot as plt
import rasterio

# from patchify import patchify, unpatchify
# from patch_msavi_thresh import patch_msavi_threshold

from gabor_texture import get_gabor_segmentation #,get_patch_area_condition,prepare_features_dataframe
from plot_utils import gabor_segmentation,soil_removed_gray_and_local_eq_gabor_mask,gabor_segmentation_mask,morphology_operations_on_gabor, single_gabor_segmentation, save_rgb_patch_png, save_mask
from vegetation_indices import VegetationIndices

from tqdm import tqdm
# %matplotlib inline

import argparse


def create_path_dict(main_path, specified_dirs=None):
    # Dictionary to store folder names and file paths
    path_dict = {}
    if specified_dirs:
        #Process only the specified directories
        for specified_dir in specified_dirs:
            folder_path = os.path.join(main_path, specified_dir)
            if os.path.isdir(folder_path):
                # Assuming the files have a specific naming pattern
                for file in os.listdir(folder_path):
                    if file.endswith(".tif"):
                        file_name = os.path.basename(file)
                        path_dict[file_name] = os.path.join(folder_path, file)
                        # path_dict[specified_dir] = os.path.join(folder_path, file)
                        break
            else:
                print(f"Directory {specified_dir} is not dir of {main_path}")
                raise NameError(specified_dir)
    else:                
    # Traverse through each folder in the given directory
        for folder_name in os.listdir(main_path):
            folder_path = os.path.join(main_path, folder_name)
        
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Look for a file inside the folder
                for file_name in os.listdir(folder_path):
                    # Construct file path
                    file_path = os.path.join(folder_path, file_name)
                
                    # Assuming we're only interested in .tif files
                    if file_name.endswith('.tif'):
                        # path_dict[file_name] = os.path.join(folder_path, file)
                        path_dict[folder_name] = file_path
                        break  # Stop after finding the first .tif file in the folder

    return path_dict

def get_ms_bands(orthomosaic_dir, prefix=""):
    # Determine where the file is with its path.
    #orthopath = f"{orthomosaic_dir}/{prefix}_2024.tif"
    orthopath = orthomosaic_dir
    if not os.path.exists(orthopath):
        print("Path doesn't exist!!!")
        raise NameError(orthopath)
    
    with rasterio.open(orthopath) as src:
        red_band = cp.array(src.read(3))
        blue_band = cp.array(src.read(1))   # I hope my notes are correct about these band numbering.
        green_band = cp.array(src.read(2))
        red_edge_band = cp.array(src.read(4))
        nir_band = cp.array(src.read(5))
    
    # stack = cp.stack((red_band, green_band, blue_band, red_edge_band, nir_band), axis=2)
    dictio = {
        'red': red_band,
        'blue': blue_band,
        'green': green_band,
        'red edge': red_edge_band,
        'nir': nir_band
    }
    return src, dictio



def get_all_bands(path_dict):
    src_dict = dict()
    band_dict = dict()

    # Loop through each key and value in path_dict
    for name, path in path_dict.items():
        # Call get_ms_bands for each item in the path_dict
        src, band_info = get_ms_bands(path)

        # Store results in src_dict and band_dict with the key as the folder name
        src_dict[name] = src
        band_dict[name] = band_info

    return src_dict, band_dict

def calculate_data_frame(ortho_path, field_name, output_directory, ndvi_or_msavi='MSAVI'):
    
    # First of all we need to create folder with fieldname, i.e Good
    output_dir = os.path.join(output_directory, field_name)
    msavi_dir = f'{output_dir}/MSAVI_Mask'
    os.makedirs(msavi_dir, exist_ok=True)
    # Const
    overlap = 0
    patch_size = 256
    #patch_thresh_area = 0.99*(patch_size**2)
    patch_thresh_area = 65300
    dpi = 64

    VI_CLASS = VegetationIndices()
    # Let's turn it into a function argument.
    soil_vi_name = "MSAVI" 
    soil_vi_func = VI_CLASS.get_vi_function(soil_vi_name)

    ndvi = 'NDVI'
    ndvi_func = VI_CLASS.get_vi_function(ndvi)

    
    columns = ['patch', 'start_vert', 'end_vert', 'start_horiz', 'end_horiz',
            'Patch_Area', 'VegetationIndex', 'VI_Thresh', 'VI_Area',
            'Pure_GaborArea', 'Pure_GaborThresh', 'LocalHistEq_GaborArea', 'LocalHistEq_GaborThresh']
    # field_orthoreflectance_dir = os.path.join(os.path.dirname(path_dict[field_name]),'OrthoReflectance/') # Eniig boliv.

    result_df = pd.DataFrame(columns = columns)
    # msavi_dir = f'/media/razydave/01DA189A333B1070/MendeFolder/Dadlaga2024BelceeriinDoroitol/Trying_to_generate_orthos/Gabor/{field_name}/MSAVI_Mask'
    # os.makedirs(msavi_dir, exist_ok=True)

    # try:
    result_df = get_gabor_segmentation(result_df, ortho_path, "", output_dir, 'Altum', field_name,
                                        soil_vi_func, soil_vi_name, ndvi_func, ndvi, save_soil_vi=True, save_gabor_fig=True,
                                        save_gabor_mask=True, overlap=overlap, patch_size=patch_size, patch_thresh_area=patch_thresh_area, dpi=dpi
                                        ,soil_vi_mask_dir=msavi_dir
                                        )
    return result_df

# The main event...conda config --set channel_priority flexible
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="""This script identifies tif files in the main directory and report. Some of the functions-written by Temuujin.
                                     The result will be saved into current working directory/CSV folder. 
                                     Example: python cli_temuujin_code.py /path/to/your/dir/one/level/above/tiffs --specify_dirs ex./Medium""", )
    parser.add_argument('directory', type=str, help='The main directory containing subfolders with .tif files')
    parser.add_argument('--specify_dirs', type=str, nargs='+', help='Specify one or more directories inside the `directory` you have provided. This will tell the algorithm to only take that folder and not recursively search for every folder inside.', required=False)
    parser.add_argument('--output_dir', type=str, required=True, help='The directory where the results will be saved')
    # Parse command line arguments

    args = parser.parse_args()
    # TODO: Use pathlib for better path handling
    main_dir = args.directory
    # output_directory = args.output_dir
    
    # output_directory = f'/media/razydave/01DA189A333B1070/MendeFolder/Dadlaga2024BelceeriinDoroitol/Trying_to_generate_orthos/Gabor'   # Change this to custom!!!
    output_dir = args.output_dir

    # Use default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), 'Gabor_Results')
        print(f"No output directory provided. Using default directory: {output_dir}")
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'Gabor_Results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # Create dictionary with folder names and file paths
    path_dict = create_path_dict(main_dir, args.specify_dirs)

    
    # Print the dictionary
    print("Path Dictionary found:")
    for key, value in path_dict.items():
        print(f'Path: ({key}: {value})')

    # src_dict, band_dict = get_all_bands(path_dict)
    # print(band_dict)
    # for key in band_dict:
    #     write_separate_bands(path_dict, band_dict[key], src_dict[key], key)
    
    # print("Values has been written to the separate band tiffs in each total of ",len(path_dict), "with Orthoreflecteance dir. ")
    
    data_dict = {}

    for key, path in path_dict.items():
        data_dict[key] = calculate_data_frame(path, key, output_dir)

    print("Finished processing dataframes. A total of ", len(data_dict), "has been processed.")
    os.makedirs(f"{output_dir}/CSV", exist_ok=True)
    for key in data_dict:
        lowercase_key = key.lower()
        data_dict[key].to_csv(f"{output_dir}/CSV/{lowercase_key}_data.csv")

    print("Finished saving them to csv in the parent folder.")

if __name__ == '__main__':
    main()

