import os
import gc
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import cupy as cp
from cupyx.scipy.ndimage import label

import cv2
from PIL import Image
import tifffile as tiff
import rasterio

from scipy import ndimage
from scipy import signal
from skimage.filters import threshold_otsu, gabor
from skimage import exposure, filters, morphology

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

#from consts import *
from vegetation_indices import VegetationIndices
from plot_utils import plot_vegetation_patch_image_enhancements, soil_removed_gray_and_local_eq_gabor_mask, save_mask, save_rgb_patch_png

import warnings
warnings.filterwarnings("ignore")

def prepare_features_dataframe(result_df, 
                               patch_number, start_vert, end_vert, start_horiz, end_horiz, patch_area,
                               vi_name, vi_area, vi_thresh,
                               og_mask_area, og_mask_thresh, local_eq_mask_area, local_eq_mask_thresh):
    
    patch_results = pd.DataFrame({"patch": [patch_number],
                                  "start_vert": [start_vert], 
                                  "end_vert": [end_vert], 
                                  "start_horiz": [start_horiz], 
                                  "end_horiz": [end_horiz],
                                  "Patch_Area": [patch_area],
                                  "VegetationIndex": [vi_name],
                                  "VI_Thresh": [vi_thresh],
                                  "VI_Area": [vi_area],
                                  "Pure_GaborArea": [og_mask_area],
                                  "Pure_GaborThresh": [og_mask_thresh],
                                  "LocalHistEq_GaborArea": [local_eq_mask_area],
                                  "LocalHistEq_GaborThresh": [local_eq_mask_thresh]})
    
    return pd.concat([result_df, patch_results])

def get_patch_area_condition(rgb_patch, red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch, 
                             patch_thresh_area):
    cond_area = None
    if rgb_patch is not None:
        rgb_patch_gray = cv2.cvtColor(rgb_patch.get(), cv2.COLOR_RGB2GRAY)
        rgb_patch_total_area = np.where(rgb_patch_gray != 0, 1, 0).sum()
        cond_area = rgb_patch_total_area

        cond = rgb_patch_total_area <= patch_thresh_area

        del rgb_patch_gray
    else:
        red_band_patch_normalzied = cv2.normalize(red_band_patch.get(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        green_band_patch_normalzied = cv2.normalize(green_band_patch.get(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        blue_band_patch_normalzied = cv2.normalize(blue_band_patch.get(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        red_edge_band_patch_normalzied = cv2.normalize(red_edge_band_patch.get(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        nir_band_patch_normalzied = cv2.normalize(nir_band_patch.get(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        red_total_area = np.where(red_band_patch_normalzied != 0, 1, 0).sum()
        green_total_area = np.where(green_band_patch_normalzied != 0, 1, 0).sum()
        blue_total_area = np.where(blue_band_patch_normalzied != 0, 1, 0).sum()
        red_edge_total_area = np.where(red_edge_band_patch_normalzied != 0, 1, 0).sum()
        nir_total_area = np.where(nir_band_patch_normalzied != 0, 1, 0).sum()

        cond_area = np.max([red_total_area, green_total_area, blue_total_area, red_edge_total_area, nir_total_area])

        cond = (red_total_area <= patch_thresh_area) or (green_total_area <= patch_thresh_area) or \
                (blue_total_area <= patch_thresh_area) or (red_edge_total_area <= patch_thresh_area) or (nir_total_area <= patch_thresh_area)
        
        
        del red_band_patch_normalzied, green_band_patch_normalzied, blue_band_patch_normalzied, red_edge_band_patch_normalzied, nir_band_patch_normalzied
    
    gc.collect()

    return cond, cond_area

def get_ms_bands(orthomosaic_dir, prefix=""):       # This function is added by Mendee # This confirms that we don't need to read bands separately.
        # Determine where the file is with its path.
        #orthopath = f"{orthomosaic_dir}/{prefix}_2024.tif"
        orthopath = orthomosaic_dir
        if not os.path.exists(orthopath):
            print("Path doesn't exist!!!")
            # print(NameError(orthopath))
            raise FileNotFoundError(f"The file {orthopath} does not exist.")
            
            
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



def get_gabor_segmentation(result_df, orthomosaic_dir: str, prefix: str, output_folder:str, drone: str, target_class,
                           soil_vi_func, soil_vi_name, vegetation_vi_func, vegetation_vi_name, 
                           save_rgb_patch = False, save_soil_vi = False, soil_vi_mask_dir = None, save_vegetation_index = False,
                           save_gabor_mask = False, save_gabor_fig = False, 
                           overlap = 0, patch_size = 512, patch_thresh_area = 10000, dpi = 64) -> pd.DataFrame:
    """This function is a big function, which is responsible for processing the given orthomosaic image
    it should return a dataframe with the following columns:
    patch, start_vert, end_vert, start_horiz, end_horiz, Patch_Area, VegetationIndex, VI_Thresh, VI_Area, Pure_GaborArea, Pure_GaborThresh, LocalHistEq_GaborArea, LocalHist
    """
    # Temuujin's code must be modified to work with Single source tiffs
    rgb_img = None
    print("from gabor -> get_gabor_segmentation says:Changes were up.")
    # red_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_red.tif'
    # green_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_green.tif'
    # blue_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_blue.tif'
    # red_edge_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_red edge.tif'
    # nir_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_nir.tif'
    
    # Agisoft Metashape and WebODM outputs
    # orthomosaic_dir
    
    # if drone == 'DJI_P4':
    #     rgb_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_group1.tif'
    #     rgb_img = tiff.imread(rgb_path)
    #     rgb_img = cp.asarray(rgb_img[:, :, :3])

    #     red_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_red.tif'
    #     green_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_green.tif'
    #     blue_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_blue.tif'
    #     red_edge_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_red edge.tif'
    #     nir_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_nir.tif'

    # with rasterio.open(red_path) as red_src:
    #     red_band = cp.array(red_src.read(1))

    # with rasterio.open(blue_path) as blue_src:
    #     blue_band = cp.array(blue_src.read(1))

    # with rasterio.open(green_path) as green_src:      # There are no paths for them
    #     green_band = cp.array(green_src.read(1))

    # with rasterio.open(red_edge_path) as red_edge_src:
    #     red_edge_band = cp.array(red_edge_src.read(1))

    # with rasterio.open(nir_path) as nir_src:
    #     nir_band = cp.array(nir_src.read(1))
    
    
    src, band_dict = get_ms_bands(orthomosaic_dir, prefix)
    
    red_band = band_dict['red']
    blue_band = band_dict['blue']
    green_band = band_dict['green']
    red_edge_band = band_dict['red edge']
    nir_band = band_dict['nir']

    height, width = cp.shape(red_band)
    num_patches_vert = (height - overlap) // (patch_size - overlap)
    num_patches_horiz = (width - overlap) // (patch_size - overlap)

    jet_cmap = LinearSegmentedColormap.from_list('jet', ['black', 'blue', 'green', 'yellow', 'red', 'white'], N = 256)

    print(f"\n{vegetation_vi_name} Vegetation Index processing started:")
    for i in tqdm(range(num_patches_vert)):
        for j in range(num_patches_horiz):
            start_vert = i * (patch_size - overlap)
            end_vert = start_vert + patch_size

            start_horiz = j * (patch_size - overlap)
            end_horiz = start_horiz + patch_size

            # Extract bands (converted to CuPy arrays)
            cond = None
            rgb_patch = None

            # if drone == 'DJI_P4':
            #     rgb_patch = rgb_img[start_vert:end_vert, start_horiz:end_horiz, :]
            
            red_band_patch = red_band[start_vert:end_vert, start_horiz:end_horiz]
            green_band_patch = green_band[start_vert:end_vert, start_horiz:end_horiz]
            blue_band_patch = blue_band[start_vert:end_vert, start_horiz:end_horiz]
            red_edge_band_patch = red_edge_band[start_vert:end_vert, start_horiz:end_horiz]
            nir_band_patch = nir_band[start_vert:end_vert, start_horiz:end_horiz]
            # Mendee added #2024.10.24 May be this is not needed.
            # if prefix == 'Worst':
            #     min_shape = np.min([red_band_patch.shape, green_band_patch.shape, blue_band_patch.shape], axis=0)
            # try:
            #     rgb_patch  = np.dstack((red_band_patch / np.max(red_band_patch),
            #                  green_band_patch / np.max(green_band_patch) , blue_band_patch / np.max(blue_band_patch)))
            # except Exception as e:
            #     print("An error occured at ", (i,j), "\n", e)
            #     print("The possible cause:", red_band_patch.shape," ", green_band_patch.shape," ", blue_band_patch.shape )       
            # Mendee end
            # Condition for trimming edges of full image
            cond, cond_area = get_patch_area_condition(rgb_patch, red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch, patch_thresh_area)

            if cond:
                print(f'[INFO]({i}, {j}) skipped because of {cond_area}')    # Mendee added for console.log
                continue
            else:
                # print(f'({i}, {j}) proceeded')
                if drone == 'Altum':
                    multistack = cp.stack([red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch], axis = 2).get()
                    multistack = np.uint8(255 * (multistack / multistack.max()))
                    rgb_patch = cp.asarray(multistack[:, :, :3])

                    del multistack
                    gc.collect()
                # Mendee Indented the code to eliminate try-except block
                # try:
                patch_number = f"patch_{i}_{j}"
                patch_filename = f"patch_{start_vert}_{end_vert}_{start_horiz}_{end_horiz}"
                
                soil_vi = None
                soil_vi_mask = None

                if save_rgb_patch:
                    rgb_patch_output_folder = os.path.join(output_folder, 'RGB_Patch')
                    rgb_patch_output_filename = f'{patch_filename}_rgb.png'
                    save_rgb_patch_png(rgb_patch, rgb_patch_output_folder, rgb_patch_output_filename)

                if save_soil_vi:
                    # Calculating and Separating Soil from Vegetation (using CuPy functions)
                    soil_vi = soil_vi_func(red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch)
                    soil_vi[~cp.isfinite(soil_vi)] = 0 #replacing infinite values with 0 added by Mendee
                    soil_vi_thresh = threshold_otsu(soil_vi)
                    # maybe constant threshold?
                    # soil_vi_thresh = cp.array(0.2);    # https://eos.com/make-an-analysis/msavi/#how-to-interpret-values:~:text=On%20our%20agriculture,cover%20the%20soil.

                    soil_vi_mask = cp.where(soil_vi > soil_vi_thresh, 1, 0)
                    soil_area = cp.where(soil_vi <= soil_vi_thresh, 1, 0).get().sum()

                    # Saving Soil Index Related Values
                    result_df = prepare_features_dataframe(result_df = result_df, patch_number = patch_number, 
                                                            start_vert = start_vert, end_vert = end_vert, 
                                                            start_horiz = start_horiz, end_horiz = end_horiz, patch_area = cond_area,
                                                            vi_name = soil_vi_name, vi_area = soil_area, vi_thresh = soil_vi_thresh.get(),
                                                            og_mask_area = np.nan, og_mask_thresh = np.nan, 
                                                            local_eq_mask_area = np.nan, local_eq_mask_thresh = np.nan)
                    soil_output_folder = soil_vi_mask_dir
                    soil_output_filename = f'{patch_filename}_{soil_vi_name}_mask.npy'
                    save_mask(soil_vi_mask.get(), soil_output_folder, soil_output_filename)
                else:
                    soil_vi_mask_filename = os.path.join(soil_vi_mask_dir, f'{patch_filename}_{soil_vi_name}_mask.npy')
                    soil_vi_mask = cp.asarray(np.load(soil_vi_mask_filename))

                # Soil Vegetation Index Masked bands (using CuPy masking)
                red_masked = red_band_patch * soil_vi_mask
                green_masked = green_band_patch * soil_vi_mask
                blue_masked = blue_band_patch * soil_vi_mask
                red_edge_masked = red_edge_band_patch * soil_vi_mask
                nir_masked = nir_band_patch * soil_vi_mask

                # Calculating given Vegetation index
                vegetation_vi = vegetation_vi_func(red_masked, green_masked, blue_masked, red_edge_masked, nir_masked)
                vegetation_vi[~cp.isfinite(vegetation_vi)] = 0  #added by Mendee
                vegetation_vi_thresh = threshold_otsu(vegetation_vi)
                vegetation_vi_mask = cp.where(vegetation_vi > vegetation_vi_thresh, 1, 0)
                vegetation_vi_mask_area = vegetation_vi_mask.get().sum()

                if save_vegetation_index:
                    vi_values_output_folder = os.path.join(output_folder, vegetation_vi_name)
                    vi_values_output_filename = f'{patch_filename}_{vegetation_vi_name}.npy'
                    save_mask(vegetation_vi.get(), vi_values_output_folder, vi_values_output_filename)

                # Gabor Texture Segmentation #########################################
                rgb_patch_gray = cv2.cvtColor(rgb_patch.get(), cv2.COLOR_RGB2GRAY)
                soil_removed_gray = (cp.asarray(rgb_patch_gray) * soil_vi_mask).get()
                soil_removed_gray[~np.isfinite(soil_removed_gray)] = 0  # added by Mendee
                #local_hist_mask

                og_mask, og_mask_thresh, og_mask_area, local_hist_mask, local_hist_thresh, local_eq_mask_area = soil_removed_gray_and_local_eq_gabor_mask(soil_removed_gray)

                if save_gabor_mask:
                    # Soil removed grayscale image gabor segmentation mask
                    og_gabor_mask_output_folder = os.path.join(output_folder, 'OriginalGrayscale_Mask')
                    og_gabor_mask_filename = f'{patch_filename}_original_gabor_mask.npy'
                    save_mask(og_mask, og_gabor_mask_output_folder, og_gabor_mask_filename)

                    # Local Histogram Equalized
                    local_eq_gabor_mask_output_folder = os.path.join(output_folder, 'LocalHistEq_Mask')
                    local_eq_gabor_mask_filename = f'{patch_filename}_local_eq_gabor_mask.npy'
                    save_mask(local_hist_mask, local_eq_gabor_mask_output_folder, local_eq_gabor_mask_filename)

                # Saving figure
                if save_gabor_fig:
                    fig_output_folder = os.path.join(output_folder, 'Original_and_Local_Hist_Equalized')
                    fig_output_filename = f"patch_{start_vert}_{end_vert}_{start_horiz}_{end_horiz}.png"
                    plot_vegetation_patch_image_enhancements(og_mask, local_hist_mask, 
                                                                soil_vi, soil_vi_mask, soil_vi_name,
                                                                vegetation_vi, rgb_patch, vegetation_vi_name, jet_cmap, 
                                                                fig_output_folder, fig_output_filename, dpi = dpi)
                
                # Saving Vegetation Index Related Values
                result_df = prepare_features_dataframe(result_df = result_df, patch_number = patch_number, 
                                                        start_vert = start_vert, end_vert = end_vert, 
                                                        start_horiz = start_horiz, end_horiz = end_horiz, patch_area = cond_area,
                                                        vi_name = vegetation_vi_name, vi_area = vegetation_vi_mask_area, vi_thresh = vegetation_vi_thresh.get(),
                                                        og_mask_area = og_mask_area, og_mask_thresh = og_mask_thresh, 
                                                        local_eq_mask_area = local_eq_mask_area, local_eq_mask_thresh = local_hist_thresh)
                
                del (soil_vi, soil_vi_mask, vegetation_vi, vegetation_vi_mask, og_mask, og_mask_area, 
                        local_hist_mask, local_eq_mask_area, og_mask_thresh, local_hist_thresh)
                gc.collect()
                # except Exception as e:
                #     print(f"An error occured at index {(i, j)}")
                #     print(e)
                #     continue
                
            del rgb_patch, red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch
            gc.collect()

    print(f"{vegetation_vi_name} Vegetation Index processing finished.")
    print("*"*64)

    del rgb_img, red_band, green_band, blue_band, red_edge_band, nir_band
    gc.collect()

    return result_df