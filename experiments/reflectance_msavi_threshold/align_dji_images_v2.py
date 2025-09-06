# This script is different from the `align_dji_images.py` because the previous version used to normalize the reflectance 
# values, which I feel little wrong. However, this version is not still that good because the MSAVI plots seem to 
# indicate that msavi has become very dull.

# This is basically just saving the images as float. Don't know which one is correct.

import cv2
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import sys

from pathlib import Path

# import argparse

## Utility functions
def convert_to_8bit(image):
    """ Convert 16-bit images to 8-bit for visualization"""
    # Check if image is 16-bit
    assert image.dtype == np.uint16, "Image must be 16-bit unsigned integer"
    # Normalize to 0-255 range
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image_8bit


# 2025.05.14 Change. Thought that reflectance images need to be converted into 0.0 to 1.0 range
def convert_to_float(image):
    """ Must be used for reflectance images only"""
    assert image.dtype == np.uint16, "Image must be 16-bit unsigned integer"
    image_float = image.astype(np.float32)
    image_float /= 65535.0  # Normalize to 0-1 range
    return image_float


def show_imgs_side_by_side(img1, img2, title1='Image 1', title2='Image 2'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2)
    ax2.axis('off')
    plt.show()

def normalize_16bit(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # scale to 0-1
    img = (img * 65535).astype(np.uint16)  # scale to 0-65535
    return img

def align_images_fallback(gray1, gray0):
    """
    Alignes image1 to an image 0 returning image1 aligned to image0 along with image0 in tuple.
    If either image1 or image0 2 dimensional image it would cause error because of the cv2.cvtColor() used in the funcion.
    """
    # Convert images to grayscale
    # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)

    # Create ORB feature detector
    orb = cv2.ORB_create(nfeatures= 2000)

    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints0, descriptors0 = orb.detectAndCompute(gray0, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors0)

    top_matches = sorted(matches, key=lambda x: x.distance)

    src_points = np.float32([keypoints1[match.queryIdx].pt for match in top_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints0[match.trainIdx].pt for match in top_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    aligned_image1 = cv2.warpPerspective(gray1, homography, (gray0.shape[1], gray0.shape[0]))
    return aligned_image1, homography

# ECC-based alignment
def align_with_ecc(reference_image, target_image,
                    warp_mode=cv2.MOTION_AFFINE,
                    number_of_iterations = 500,
                    termination_eps = 1e-5):
    sz = reference_image.shape
    warp_matrix = None
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(reference_image, target_image, warp_matrix, warp_mode, criteria)
    except cv2.error as e:
        print(f"Error in ECC alignment: {e}. Running ORB alignment instead.")
        # convert images to 8bit
        reference_image = convert_to_8bit(reference_image)
        target_image = convert_to_8bit(target_image)
        aligned, target_image = align_images_fallback(reference_image, target_image)
        return aligned, warp_matrix
        
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        aligned = cv2.warpPerspective(target_image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        aligned = cv2.warpAffine(target_image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return aligned, warp_matrix

def align_with_warp_matrix(reference_image, target_image, warp_matrix):
    """
    Aligns the target image to the reference image using the provided warp matrix.
    
    Parameters:
    -----------
    reference_image : numpy.ndarray
        The reference image to which the target image will be aligned.
    target_image : numpy.ndarray
        The target image to be aligned.
    warp_matrix : numpy.ndarray
        The warp matrix obtained from ECC or other methods.
    
    Returns:
    --------
    numpy.ndarray
        Aligned target image.
    """
    assert target_image.dtype == np.float32, "Target image must be of type float32"
    fill_value = -9999.0
    sz = reference_image.shape
    aligned = cv2.warpAffine(target_image, 
                             warp_matrix,
                            (sz[1], sz[0]), 
                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=fill_value)
    aligned = np.where(aligned == -9999.0, np.nan, aligned)
    return aligned

def calculate_ndvi(nir_band, red_band):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from NIR and Red bands.
    
    Parameters:
    -----------
    nir_band : numpy.ndarray
        Near-infrared band image
    red_band : numpy.ndarray
        Red band image
    
    Returns:
    --------
    numpy.ndarray
        NDVI values ranging from -1 to 1
    numpy.ndarray
        NDVI values scaled to 0-255 for visualization
    """
    # Convert to float to allow for division
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    
    # Calculate NDVI (NIR - Red) / (NIR + Red)
    # Avoid division by zero
    denominator = nir + red
    ndvi = np.zeros_like(nir)
    valid_pixels = denominator > 0
    # ndvi[valid_pixels] = (nir[valid_pixels] - red[valid_pixels]) / denominator[valid_pixels]
    ndvi = (nir - red) / denominator

    assert valid_pixels.sum() > 0.95 * np.prod(valid_pixels.shape), "Division by zero in NDVI calculation"
    
    # Scale NDVI to 0-255 for visualization
    ndvi_scaled = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    
    return ndvi, ndvi_scaled

def calculate_msavi(nir_band, red_band):
    """
    Calculate the Modified Soil Adjusted Vegetation Index (MSAVI).
    
    MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
    
    This vegetation index minimizes soil background effects and is useful
    in areas with partial vegetation cover.
    
    Parameters:
    -----------
    nir_band : numpy.ndarray
        Near-infrared band image
    red_band : numpy.ndarray
        Red band image
    
    Returns:
    --------
    numpy.ndarray
        MSAVI values ranging from -1 to 1
    numpy.ndarray
        MSAVI values scaled to 0-255 for visualization
    """
    # Convert to float to allow for division and square root
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    
    # Calculate MSAVI using the formula
    # MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
    
    term1 = 2 * nir + 1
    term2 = np.sqrt(np.square(2 * nir + 1) - 8 * (nir - red))
    msavi = (term1 - term2) / 2
    
    # Scale MSAVI to 0-255 for visualization
    # MSAVI typically ranges from -1 to 1
    msavi_scaled = ((msavi + 1) / 2 * 255).astype(np.uint8)
    
    return msavi, msavi_scaled

def plot_index(ndvi, index: str='NDVI', cmap: str='RdYlGn', domain: str = ""):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a normalized color scale that handles the range of NDVI values properly
    norm = Normalize(vmin=-1, vmax=1)
    ndvi_plot = ax.imshow(ndvi, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = fig.colorbar(ndvi_plot, ax=ax, shrink=0.8)
    cbar.set_label(f'{index} Value')
    
    # Add title and turn off axis
    ax.set_title(f"{index} with {cmap.upper()} Colormap at {domain}", fontsize=16)
    ax.axis('off')
    return fig
# utility function for msavi and ndvi
def nan_outside_range(array, min_val, max_val):
    """
    Replace values in the input array that are outside the [min_val, max_val] range with np.nan.

    Parameters:
        array (np.ndarray): Input array.
        min_val (float): Minimum allowed value.
        max_val (float): Maximum allowed value.

    Returns:
        np.ndarray: Array with values outside the range replaced by np.nan.
    """
    array = array.astype(float)  # Ensure we can assign np.nan
    outside_range = (array < min_val) | (array > max_val)
    array[outside_range] = np.nan
    return array

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    image_path = Path(input_dir)
    image_names = (image_path).glob('DJI_0*.TIF')
    image_names = sorted(list(image_names))
    band_names = ['Blue', 'Green', 'Red', 'Red Edge', 'NIR'] # in order
    print("Found following images:")
    
    for index, image in enumerate(image_names):
        print(f"{band_names[index]} [{index}]: {image}")
    
    # Open and read images
    images_raw = [cv2.imread(str(image), cv2.IMREAD_UNCHANGED) for image in image_names]
    images_normed = [normalize_16bit(image) for image in images_raw]
    images = [convert_to_8bit(image) for image in images_normed]
    blue = images[0]
    green = images[1]
    red = images[2]
    #image1 = convert_to_8bit(images[4])
    red_edge = images[3]
    nir = images[4]

    print(f"nir dtype: {nir.dtype}, red dtype: {red.dtype}")

    print("Using red as reference image: ", image_names[2])
    _, blue_warp = align_with_ecc(red, blue)
    _, green_warp = align_with_ecc(red, green)
    _, red_warp = align_with_ecc(red, red_edge)
    _, nir_warp = align_with_ecc(red, nir)
    # Align reflectance bands according to the warp matrix of each band

    images_float = [convert_to_float(image) for image in images_raw]
    blue_float = images_float[0]
    green_float = images_float[1]
    red_float = images_float[2]
    #image1 = convert_to_8bit(images[4])
    red_edge_float = images_float[3]
    nir_float = images_float[4]

    blue_aligned_float = align_with_warp_matrix(red_float, blue_float, blue_warp)
    green_aligned_float = align_with_warp_matrix(red_float, green_float, green_warp)
    red_edge_aligned_float = align_with_warp_matrix(red_float, red_edge_float, red_warp)
    nir_aligned_float = align_with_warp_matrix(red_float, nir_float, nir_warp)

    aligned_bands = [blue_aligned_float, green_aligned_float, red_float, red_edge_aligned_float, nir_aligned_float]
    valid_mask = np.all([~np.isnan(band) for band in aligned_bands], axis=0)
    print(f"Valid percentage: {np.sum(valid_mask) / np.prod(valid_mask.shape) * 100:.2f}%") 
    # nir_aligned_float = convert_to_float(nir_aligned) # change 2025.05.14
    # red_float = convert_to_float(red) # change 2025.05.14
    ndvi, ndvi_scaled = calculate_ndvi(nir_aligned_float, red_float)
    ndvi[~valid_mask] = np.nan
    # nir_aligned_float_masked = valid_mask * nir_aligned_float
    msavi, msavi_scaled = calculate_msavi(nir_aligned_float, red_float)
    msavi[~valid_mask] = np.nan
    # Plot NDVI
    # fig = plot_index(ndvi, index='NDVI', cmap='RdYlGn',domain=f" {image_path.parent.name}")

    
    fig = plot_index(ndvi, index='NDVI', cmap='jet',domain=f" {image_path.parent.name}")
    # Plot MSAVI
    fig_msavi = plot_index(msavi, index='MSAVI', cmap='jet',domain=f" {image_path.parent.name}")
    
    
    # Save NDVI plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Saving aligned bands to {output_path} directory...")
    cv2.imwrite(str(output_path / "red_reference.tif"), red_float)
    band_names = ['Blue', 'Green', 'Red Edge', 'NIR'] # in order
    for index, (band_name, band) in enumerate(zip(band_names, [blue_aligned_float, green_aligned_float, red_edge_aligned_float, nir_aligned_float])):
        output_name = output_path / f"{band_name.lower()}_aligned.tif"
        cv2.imwrite(str(output_name), band)
        print(f"Saved {band_names[index]} to: {output_name}")

    # Save NDVI plot
    cv2.imwrite(str(output_path / "ndvi.tif"), ndvi)
    cv2.imwrite(str(output_path / "msavi.tif"), msavi)
    cv2.imwrite(str(output_path / "ndvi_scaled.tif"), ndvi_scaled)
    cv2.imwrite(str(output_path / "msavi_scaled.tif"), msavi_scaled)


    fig.savefig(output_path / "ndvi_plot.png")
    fig_msavi.savefig(output_path / "msavi_plot.png")
    print(f"NDVI plot saved to: {output_path}, MSAVI plot saved to : {output_path}")
    plt.show()

if __name__ == "__main__":
    main()