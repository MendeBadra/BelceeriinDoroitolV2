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

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore")


def get_img_patch(orthomosaic_dir, reflectance_dir, prefix, start_vert, end_vert, start_horiz, end_horiz):
    """
    Get a sample patch of orthomosaic and reflectance ms bands given by the location of the patch
    This makes things easier to get a sample overview of a given orthophoto without loading the whole thing. This function uses rasterio library.
    
    Parameters:
        orthomosaic_dir (string): The directory in which the orthomosaic is located.
        reflectance_dir (string): The directory in which the ms bands are located.
        start_vert (int): start from where in the big orthomosaic vertically
        end_vert (int): end where in the big ortho vertically
        start_horiz (int): start form where in the big orthomosaic horizontally
        end_horiz (int): end where in the big ortho horizontally
    
    Returns: 
    A tuple of cupy arrays of rgb patch and other 5 ms bands.
    
    Examples:
    >>> rgb, red, green, blue, redge, nir = get_patch("../OrthosNew/DJI_P4/Best/Orthomosaic", "../OrthosNew/DJI_P4/Best/Reflectance", 3584, 3584+256, 5376, 5376+256 )
     
    Doc written by Mende
    """
    rgb_path = f'{orthomosaic_dir}/{prefix}_transparent_mosaic_group1.tif'
    red_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_red.tif'
    green_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_green.tif'
    blue_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_blue.tif'
    red_edge_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_red edge.tif'
    nir_path = f'{reflectance_dir}/{prefix}_transparent_reflectance_nir.tif'

    rgb_img = tiff.imread(rgb_path)
    rgb_img = cp.asarray(rgb_img[:, :, :3])

    with rasterio.open(red_path) as red_src:
        red_band = cp.array(red_src.read(1))

    with rasterio.open(blue_path) as blue_src:
        blue_band = cp.array(blue_src.read(1))

    with rasterio.open(green_path) as green_src:
        green_band = cp.array(green_src.read(1))

    with rasterio.open(red_edge_path) as red_edge_src:
        red_edge_band = cp.array(red_edge_src.read(1))

    with rasterio.open(nir_path) as nir_src:
        nir_band = cp.array(nir_src.read(1))

    rgb_patch = rgb_img[start_vert:end_vert, start_horiz:end_horiz, :]
    red_band_patch = red_band[start_vert:end_vert, start_horiz:end_horiz]
    green_band_patch = green_band[start_vert:end_vert, start_horiz:end_horiz]
    blue_band_patch = blue_band[start_vert:end_vert, start_horiz:end_horiz]
    red_edge_band_patch = red_edge_band[start_vert:end_vert, start_horiz:end_horiz]
    nir_band_patch = nir_band[start_vert:end_vert, start_horiz:end_horiz]

    del rgb_img, red_band, green_band, blue_band, red_edge_band, nir_band
    gc.collect()

    return rgb_patch, red_band_patch, green_band_patch, blue_band_patch, red_edge_band_patch, nir_band_patch

def plot_soil_vegetation_patch(rgb_patch, soil_vi, soil_vi_thresh, soil_vi_mask, soil_vi_name, cmap):
    """
    Plots the soil vegetation patches which is given.
    
    Parameters:
        rgb_patch (array): Rgb image sample.
        soil_vi (array): Vegetation index calculated to form an array of value.
        soil_vi_thresh : threshold_otsu output
        soil_vi_mask (array): binary mask 
        soil_vi_name (string): Name of the vegetation index used. Ex: 'MSAVI', 'NDVI'
        cmap (obj) : LinearSegmentedColormap.from_list (Don't know yet!)
    
    Returns:
        matplotlib.plot stuff
    
    Doc written by: Mende
    """
    soil_vi_np = soil_vi.get()
    soil_vi_thresh_np = soil_vi_thresh.get()
    soil_vi_mask_np = (255 * soil_vi_mask).get().astype(np.uint8)
    soil_vi_masked = (soil_vi * soil_vi_mask).get()

    vmin = soil_vi_np.min()
    vmax = soil_vi_np.max()
    centered_log_norm = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                          vmin = vmin, vmax = vmax, base = 10)

    masked_vmin = soil_vi_masked.min()
    masked_vmax = soil_vi_masked.max()
    centered_log_norm_masked = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                                 vmin = masked_vmin, vmax = masked_vmax, base = 10)

    fig, ((ax1, ax2, ax3), 
          (ax4, ax5, ax6)) = plt.subplots(2, 3, num = 1, clear = True, figsize = (16, 6))

    soil_vi_cbar_imshow = ax1.imshow(soil_vi_np, norm = centered_log_norm, cmap = cmap)
    soil_vi_hist_imshow = ax2.hist(soil_vi_np.ravel(), log = True, density = True, bins = 256)
    soil_vi_hist_thresh_imshow = ax2.axvline(soil_vi_thresh_np, color = 'r')

    rgb_patch_imshow = ax3.imshow(rgb_patch.get())
    rgb_patch_masked = ax6.imshow((rgb_patch * np.expand_dims(soil_vi_mask, axis = -1)).get())

    soil_vi_cbar_masked  = ax4.imshow(soil_vi_masked, norm = centered_log_norm_masked, cmap = cmap)
    soil_vi_mask_imshow = ax5.imshow(soil_vi_mask_np, cmap = 'gray')

    ax1.set_title(f"{soil_vi_name}")
    ax2.set_title(f"{soil_vi_name} Histogram")
    ax3.set_title("RGB Patch")
    ax4.set_title(f"{soil_vi_name} Masked using Threshold")
    ax5.set_title(f"{soil_vi_name} Binary Mask")
    ax6.set_title("RGB Patch soil masked")

    ax1.axis('off')
    ax2.spines[['top', 'right']].set_visible(False)
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')

    fig.tight_layout()

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(soil_vi_cbar_imshow, cax = cax1, orientation = 'vertical')


    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(soil_vi_cbar_masked, cax = cax2, orientation = 'vertical')

    del (fig, ax1, ax2, ax3, ax4, ax5, ax6, 
        soil_vi_cbar_imshow, soil_vi_hist_imshow, rgb_patch_imshow, soil_vi_hist_thresh_imshow, soil_vi_cbar_masked, soil_vi_mask_imshow, rgb_patch_masked)
    gc.collect()

def plot_vegetation_patch(vegetation_vi, vegetation_vi_thresh, vegetation_vi_mask, vegetation_vi_name, cmap):
    vegetation_index_np = vegetation_vi.get()
    vegetation_index_thresh_np = vegetation_vi_thresh.get()
    vegetation_index_mask_np = (255 * vegetation_vi_mask).get().astype(np.uint8)
    vegetation_index_masked = (vegetation_vi * vegetation_vi_mask).get()

    vmin = vegetation_index_np.min()
    vmax = vegetation_index_np.max()
    centered_log_norm = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                          vmin = vmin, vmax = vmax, base = 10)
    
    masked_vmin = vegetation_index_masked.min()
    masked_vmax = vegetation_index_masked.max()
    centered_log_norm_masked = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                                 vmin = masked_vmin, vmax = masked_vmax, base = 10)

    fig, ((ax1, ax2), 
          (ax3, ax4)) = plt.subplots(2, 2, num = 1, clear = True, figsize = (13, 6))

    vegetation_vi_cbar_imshow = ax1.imshow(vegetation_index_np, norm = centered_log_norm, cmap = cmap)
    vegetation_vi_hist_imshow = ax2.hist(vegetation_index_np.ravel(), log = True, density = True, bins = 256)
    vegetation_vi_hist_thresh_imshow = ax2.axvline(vegetation_index_thresh_np, color = 'r')
    vegetation_vi_cbar_masked  = ax3.imshow(vegetation_index_masked, norm = centered_log_norm_masked, cmap = cmap)
    vegetation_vi_mask_imshow = ax4.imshow(vegetation_index_mask_np, cmap = 'gray')

    ax1.set_title(f"{vegetation_vi_name}")
    ax2.set_title(f"{vegetation_vi_name} Histogram")
    ax3.set_title(f"{vegetation_vi_name} Masked using Threshold")
    ax4.set_title(f"{vegetation_vi_name} Binary Mask")

    ax1.axis('off')
    ax2.spines[['top', 'right']].set_visible(False)
    ax3.axis('off')
    ax4.axis('off')
    
    fig.tight_layout()
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(vegetation_vi_cbar_imshow, cax = cax1, orientation = 'vertical')


    divider2 = make_axes_locatable(ax3)
    cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(vegetation_vi_cbar_masked, cax = cax2, orientation = 'vertical')

    del (fig, ax1, ax2, ax3, ax4, 
         vegetation_vi_cbar_imshow, vegetation_vi_hist_imshow, vegetation_vi_hist_thresh_imshow, vegetation_vi_cbar_masked, vegetation_vi_mask_imshow)
    gc.collect()

def plot_vegetation_patch_custom_mask(vegetation_vi, rgb_patch, mask, vegetation_vi_name, cmap, 
                                      save_fig = False, output_folder = None, output_filename = None, dpi = 80):
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok = True)
        output_filename = os.path.join(output_folder, output_filename)

    vegetation_index_np = vegetation_vi.get()
    rgb_patch_np = rgb_patch.get()
    rgb_patch_masked = (rgb_patch * np.expand_dims(mask, axis = -1)).get()

    vmin = vegetation_index_np.min()
    vmax = vegetation_index_np.max()
    centered_log_norm = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                          vmin = vmin, vmax = vmax, base = 10)
    fig, ((ax1, ax2), 
          (ax3, ax4)) = plt.subplots(2, 2, num = 1, clear = True, figsize = (13, 6))

    vegetation_vi_cbar_imshow = ax1.imshow(vegetation_index_np, norm = centered_log_norm, cmap = cmap)
    rgb_patch_imshow = ax2.imshow(rgb_patch_np)
    mask_imshow = ax3.imshow(mask.get(), cmap = 'gray')
    masked_rgb_imshow = ax4.imshow(rgb_patch_masked)
    
    ax1.set_title(f"{vegetation_vi_name}")
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    
    fig.tight_layout()
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(vegetation_vi_cbar_imshow, cax = cax1, orientation = 'vertical')

    if save_fig:
        fig.savefig(output_filename, dpi = dpi)

    del (fig, ax1, ax2, ax3, ax4, 
         vegetation_vi_cbar_imshow, rgb_patch_imshow, mask_imshow, masked_rgb_imshow)
    gc.collect()

def single_gabor_segmentation(image, frequencies = [0.1, 0.5], angles = [0, np.pi/2]):
    segmented_image = np.zeros_like(image)
    
    # Iterate over frequencies and angles
    for frequency in frequencies:
        for angle in angles:
            # Create Gabor filter
            gabor_filter_real, _ = gabor(image, frequency = frequency, theta = angle)
            # Update segmented image using Gabor response
            segmented_image += gabor_filter_real

    threshold = threshold_otsu(segmented_image)
    below_threshold_mask = np.where(segmented_image < threshold, 1, 0)
    below_threshold_image = image * below_threshold_mask

    return below_threshold_image

def gabor_segmentation(images, frequencies = [0.1, 0.5], angles = [0, np.pi/2]):
    """Perform Gabor texture segmentation."""
    all_segmented_image = []
    all_image_thresh = []

    for image in images:
        segmented_image = np.zeros_like(image)
        # Iterate over frequencies and angles
        for frequency in frequencies:
            for angle in angles:
                # Create Gabor filter
                gabor_filter_real, _ = gabor(image, frequency = frequency, theta = angle)
                # Update segmented image using Gabor response
                segmented_image += gabor_filter_real

        threshold = threshold_otsu(segmented_image)
        below_threshold_mask = np.where(segmented_image < threshold, 1, 0)
        below_threshold_image = image * below_threshold_mask

        all_segmented_image.append(below_threshold_image)
        all_image_thresh.append(threshold)
    
    return all_segmented_image, all_image_thresh

def morphology_operations_on_gabor(all_segmented_image, size_1 = 3, size_2 = 5, size_3 = 4):
    morph_imgs = []

    for img in all_segmented_image:
        kernel = morphology.disk(size_1)
        closed_image = morphology.closing(img, kernel)

        kernel = morphology.disk(size_2)
        opened_image = morphology.opening(closed_image, kernel)

        kernel = morphology.disk(size_3)
        closed_image = morphology.closing(opened_image, kernel)

        morph_imgs.append(closed_image)

    return morph_imgs

def gabor_segmentation_mask(images):
    all_segmented_image, all_image_thresh = gabor_segmentation(images, frequencies = [0.1, 0.5], angles = [0, np.pi/2])
    all_cleaned_image = morphology_operations_on_gabor(all_segmented_image, size_1 = 3, size_2 = 5, size_3 = 4)

    binary_mask = []
    for clean_seg in all_cleaned_image:
        tmp = cp.asarray(np.where(clean_seg != 0, 1, 0))
        binary_mask.append(tmp)

    del all_cleaned_image, all_segmented_image
    gc.collect()

    return binary_mask, all_image_thresh

def soil_removed_gray_and_local_eq_gabor_mask(soil_removed_gray):
    # Local Histogram Equalization
    local_hist_equalized = exposure.equalize_adapthist(soil_removed_gray, clip_limit = 0.03)
    # Gabor texture segmentation binary mask
    binary_mask, all_image_thresh = gabor_segmentation_mask([soil_removed_gray, local_hist_equalized])
    
    og_mask, local_hist_mask = binary_mask
    og_mask_thresh, local_hist_thresh = all_image_thresh

    # Compute area
    og_mask_area = og_mask.get().sum()
    local_eq_mask_area = local_hist_mask.get().sum()

    del binary_mask, all_image_thresh
    gc.collect()

    return og_mask, og_mask_thresh, og_mask_area, local_hist_mask, local_hist_thresh, local_eq_mask_area

def save_mask(mask, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok = True)
    output_filename = os.path.join(output_folder, output_filename)

    np.save(output_filename, mask)


def save_rgb_patch_png(rgb_patch, output_folder, output_filename):
    os.makedirs(output_folder, exist_ok = True)
    output_filename = os.path.join(output_folder, output_filename)

    img = Image.fromarray(np.uint8(np.clip(rgb_patch.get(), 0, 255)))
    img.save(output_filename, format = "PNG")

    del img
    gc.collect()

def plot_vegetation_patch_image_enhancements(og_mask, local_hist_mask, 
                                             soil_vi, soil_vi_mask, soil_vi_name,
                                             vegetation_vi, rgb_patch, vegetation_vi_name, cmap, 
                                             output_folder, output_filename, dpi = 80):
    os.makedirs(output_folder, exist_ok = True)
    output_filename = os.path.join(output_folder, output_filename)

    soil_vi_np = soil_vi.get()
    vmin1 = soil_vi_np.min()
    vmax1 = soil_vi_np.max()
    centered_log_norm1 = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                           vmin = vmin1, vmax = vmax1, base = 10)
    soil_vi_mask_np = (255 * soil_vi_mask).get().astype(np.uint8)

    vegetation_index_np = vegetation_vi.get()
    vmin = vegetation_index_np.min()
    vmax = vegetation_index_np.max()
    centered_log_norm = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                          vmin = vmin, vmax = vmax, base = 10)

    fig, ((ax1, ax2, ax3, ax4), 
          (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, num = 1, clear = True, figsize = (24, 12))
    
    soil_vi_cbar_imshow = ax1.imshow(soil_vi_np, norm = centered_log_norm1, cmap = cmap)
    soil_vi_binary_mask_imshow = ax5.imshow(soil_vi_mask_np, cmap = 'gray')
    
    vegetation_vi_cbar_imshow = ax2.imshow(vegetation_index_np, norm = centered_log_norm, cmap = cmap)
    rgb_patch_imshow = ax6.imshow(rgb_patch.get())

    og_mask_imshow = ax3.imshow(og_mask.get(), cmap = 'gray')
    og_masked_imshow = ax7.imshow((rgb_patch * np.expand_dims(og_mask, axis = -1)).get())

    local_eq_mask_imshow = ax4.imshow(local_hist_mask.get(), cmap = 'gray')
    local_eq_masked_imshow = ax8.imshow((rgb_patch * np.expand_dims(local_hist_mask, axis = -1)).get())

    ax1.set_title(soil_vi_name)
    ax1.set_title(f"{soil_vi_name} Binary Mask")
    
    ax2.set_title(vegetation_vi_name)
    ax6.set_title("RGB Patch")

    ax3.set_title("Soil removed grayscale mask")
    ax4.set_title("Histogram Local Equalization")

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(soil_vi_cbar_imshow, cax = cax1, orientation = 'vertical')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(vegetation_vi_cbar_imshow, cax = cax2, orientation = 'vertical')
    
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(output_filename, dpi = dpi)

    del (fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8,
         soil_vi_np, soil_vi_mask_np, soil_vi_cbar_imshow, soil_vi_binary_mask_imshow,
         vegetation_index_np, vegetation_vi_cbar_imshow, rgb_patch_imshow,
         og_mask_imshow, og_masked_imshow, local_eq_mask_imshow, local_eq_masked_imshow)
    gc.collect()

def display_image_enhancement(vegetation_vi, soil_removed_gray, vegetation_vi_name, cmap, figsize = (14, 10)):
    vegetation_index_np = vegetation_vi.get()

    vmin = vegetation_index_np.min()
    vmax = vegetation_index_np.max()
    centered_log_norm = colors.SymLogNorm(linthresh = 0.03, linscale = 0.03, 
                                          vmin = vmin, vmax = vmax, base = 10)

    # Smoothing and Sharpening
    smoothed = filters.gaussian(soil_removed_gray, sigma=1)
    sharpened = exposure.rescale_intensity(soil_removed_gray + 1.5 * (soil_removed_gray - smoothed))

    # Spatial Domain Filtering
    median_filtered = ndimage.median_filter(soil_removed_gray, size=3)

    # Frequency Domain Processing
    frequencies = np.fft.fft2(soil_removed_gray)
    frequencies_shifted = np.fft.fftshift(frequencies)
    magnitude_spectrum = 20 * np.log(np.abs(frequencies_shifted))

    # Frequency Domain Processing (High Pass Filtering)
    high_pass_filtered = np.abs(frequencies_shifted) - ndimage.gaussian_filter(np.abs(frequencies_shifted), sigma=20)

    # Frequency Domain Processing (Low Pass Filtering)
    low_pass_filtered = ndimage.gaussian_filter(soil_removed_gray, sigma=10)

    # Morphological operation to fill black areas (Dilation)
    #filled_image = morphology.binary_dilation(soil_removed_gray)
    closed_image = morphology.closing(soil_removed_gray)

    # Opening operation to remove white or gray regions within black
    opened_image = morphology.opening(soil_removed_gray)

    # Edge Enhancement
    sobelx = filters.sobel_h(soil_removed_gray)
    sobely = filters.sobel_v(soil_removed_gray)
    edge_enhanced = np.sqrt(sobelx**2 + sobely**2)

    # Contrast Enhancement
    contrast_stretched = exposure.rescale_intensity(soil_removed_gray)

    # Histogram Equalization
    hist_equalized = exposure.equalize_hist(soil_removed_gray)

    # Local Histogram Equalization
    local_hist_equalized = exposure.equalize_adapthist(soil_removed_gray, clip_limit=0.03)

    # Deblurring (example, you may need to adjust parameters)
    deblurred = filters.gaussian(soil_removed_gray, sigma=1)

    # Displaying the results
    fig, axes = plt.subplots(3, 5, num=1, clear=True, figsize=figsize)
    fig.suptitle("Image Enhancement Techniques", fontsize=16)

    # Original Grayscale
    axes[0, 0].imshow(soil_removed_gray, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 0].axis('off')

    # Contrast Stretched
    axes[0, 1].imshow(contrast_stretched, cmap='gray')
    axes[0, 1].set_title("Contrast Stretched")
    axes[0, 1].axis('off')

    # Global Histogram Equalization
    axes[0, 2].imshow(hist_equalized, cmap='gray')
    axes[0, 2].set_title("Global Hist Equalization")
    axes[0, 2].axis('off')

    # Local Histogram Equalization
    axes[0, 3].imshow(local_hist_equalized, cmap='gray')
    axes[0, 3].set_title("Local Hist Equalization")
    axes[0, 3].axis('off')

    # Vegetation index
    vegetation_vi_cbar_imshow = axes[0, 4].imshow(vegetation_index_np, norm = centered_log_norm, cmap = cmap)
    axes[0, 4].set_title(f"{vegetation_vi_name}")

    divider1 = make_axes_locatable(axes[0, 4])
    cax1 = divider1.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(vegetation_vi_cbar_imshow, cax = cax1, orientation = 'vertical')

    # Smoothing
    axes[1, 0].imshow(smoothed, cmap='gray')
    axes[1, 0].set_title("Smoothed")
    axes[1, 0].axis('off')

    # Sharpened
    axes[1, 1].imshow(sharpened, cmap='gray')
    axes[1, 1].set_title("Sharpened")
    axes[1, 1].axis('off')

    # Median Filtered
    axes[1, 2].imshow(median_filtered, cmap='gray')
    axes[1, 2].set_title("Median Filtered")
    axes[1, 2].axis('off')

    # High Pass Filtering (Example: using Sobel)
    axes[1, 3].imshow(high_pass_filtered, cmap='gray')
    axes[1, 3].set_title("High Pass Filtering")
    axes[1, 3].axis('off')

    # Low Pass Filtering (Example: Gaussian Blur)
    axes[1, 4].imshow(low_pass_filtered, cmap='gray')
    axes[1, 4].set_title("Low Pass Filtering")
    axes[1, 4].axis('off')

    # Frequency Domain Processing (Magnitude Spectrum)
    axes[2, 0].imshow(magnitude_spectrum, cmap='gray')
    axes[2, 0].set_title("Frequency Domain (Magnitude Spectrum)")
    axes[2, 0].axis('off')

    # Edge Enhancement
    axes[2, 1].imshow(edge_enhanced, cmap='gray')
    axes[2, 1].set_title("Edge Enhancement")
    axes[2, 1].axis('off')

    # Deblurred
    axes[2, 2].imshow(deblurred, cmap='gray')
    axes[2, 2].set_title("Deblurred")
    axes[2, 2].axis('off')

    # Morphological Operation (Closed Image)
    axes[2, 3].imshow(closed_image, cmap='gray')
    axes[2, 3].set_title("Closed Image")
    axes[2, 3].axis('off')

    # Morphological Operation (Opened Image)
    axes[2, 4].imshow(opened_image, cmap='gray')
    axes[2, 4].set_title("Opened Image")
    axes[2, 4].axis('off')

    fig.tight_layout()
    del fig, axes
    gc.collect()

def test_morphological_operations(soil_removed_gray):
    fig, axes = plt.subplots(4, 8, num=1, clear=True, figsize=(16, 8))
    fig.suptitle("Morphological Operations Test", fontsize=16)

    # Closing with circular kernel
    for i, size in enumerate(range(1, 9)):
        kernel = morphology.disk(size)
        closed_image = morphology.closing(soil_removed_gray, kernel)
        axes[0, i].imshow(closed_image, cmap='gray')
        axes[0, i].set_title(f"Closing (Disk, {size}x{size})")
        axes[0, i].axis('off')

    # Closing with square kernel
    for i, size in enumerate(range(1, 9)):
        kernel = morphology.square(size)
        closed_image = morphology.closing(soil_removed_gray, kernel)
        axes[1, i].imshow(closed_image, cmap='gray')
        axes[1, i].set_title(f"Closing (Square, {size}x{size})")
        axes[1, i].axis('off')

    # Opening with circular kernel
    for i, size in enumerate(range(1, 9)):
        kernel = morphology.disk(size)
        opened_image = morphology.opening(soil_removed_gray, kernel)
        axes[2, i].imshow(opened_image, cmap='gray')
        axes[2, i].set_title(f"Opening (Disk, {size}x{size})")
        axes[2, i].axis('off')

    # Opening with square kernel
    for i, size in enumerate(range(1, 9)):
        kernel = morphology.square(size)
        opened_image = morphology.opening(soil_removed_gray, kernel)
        axes[3, i].imshow(opened_image, cmap='gray')
        axes[3, i].set_title(f"Opening (Square, {size}x{size})")
        axes[3, i].axis('off')

    fig.tight_layout()
    del fig, axes
    gc.collect()
