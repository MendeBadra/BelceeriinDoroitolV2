## Script for the coregistration of multispectral images form the DJI Mavic 3M
## Using CV2 for coregistration/alignment; handling tif metadata with piexif

import os
import sys
import cv2
import numpy as np
from PIL import Image
from glob import glob
import piexif

if __name__ == "__main__":
  input_dir = sys.argv[1]

def load_image(filename):
    image = np.asarray(Image.open(filename))
    # metadata
    exif_dict = piexif.load(filename)
    metadata = piexif.dump(exif_dict) # dump as bytes
    return image, metadata

def save_image(filename, image, metadata):
    # Save the image with the original metadata
    image_arr = Image.fromarray(image)
    image_arr.save(filename, "tiff", exif=metadata)


"""using cv2.MOTION_HOMOGRAPHY"""
def align_images(image1, image2):
    # Convert images to float32
    image1_float = np.float32(image1)
    image2_float = np.float32(image2)

    # Size of image1
    sz = image1_float.shape
    
    """ Motion model HOMOGRAPHY:"""
    warp_mode = cv2.MOTION_HOMOGRAPHY
    # Define 3x3 matrix and initialize the matrix to identity
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    
    """ Motion model AFFINE:"""
    # warp_mode = cv2.MOTION_AFFINE
    # # Define 2x3 matrix and initialize the matrix to identity
    # warp_matrix = np.eye(2, 3, dtype=np.float32) # use this with AFFINE model above

    # Define the number of iterations and the termination epsilon
    # working settings for   most   images: 400, 1e-5 # try to optimize for speed while keeping accuracy good enough
    # working settings for   "all"" images: 500???, 1e-6??? # try to optimize for speed while keeping accuracy good enough
    number_of_iterations = 1000 #5000
    termination_eps = 1e-7      #1e-10
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    cc, warp_matrix = cv2.findTransformECC(image1_float, image2_float, warp_matrix, warp_mode, criteria)
    
    # Use warpPerspective
    image2_aligned = cv2.warpPerspective(image2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)    # for homography transformation
    # image2_aligned = cv2.warpAffine(image2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)       # for affine transformation
    
    return image2_aligned

def process_directory(input_dir):
    # Create the output directory if it doesn't exist
    output_directory = os.path.join(input_dir, 'aligned')
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all green band images (reference images)
    green_images = glob(os.path.join(input_dir, '*_G.TIF'))
    
    for green_image_path in green_images:
        # Extract the image ID
        base_name = os.path.basename(green_image_path)
        image_id = base_name.split('_')[-3]
        
        # Load the green band image (reference image)
        imageG, metadataG = load_image(green_image_path)
        
        # Initialize a dictionary to store aligned images and metadata
        aligned_images = {'G': (imageG, metadataG)}
        
        # Process each spectral band
        for band in ['R', 'RE', 'NIR']:
            band_image_path = green_image_path.replace('_G.TIF', f'_{band}.TIF')
            if os.path.exists(band_image_path):
                # Load the current band image
                image_band, metadata_band = load_image(band_image_path)
                
                # Align the current band image to the green band image
                aligned_band_image = align_images(imageG, image_band)
                
                # Store the aligned image and metadata
                aligned_images[band] = (aligned_band_image, metadata_band)
        
        # Save all aligned images, including the reference green band image
        for band, (aligned_image, metadata) in aligned_images.items():
            aligned_image_path = os.path.join(output_directory, f'{base_name.replace("_G.TIF", f"_{band}.TIF")}')
            save_image(aligned_image_path, aligned_image, metadata)
    
    print("Batch image registration completed and saved successfully.")

if __name__ == "__main__":
    # input_dir = "C:\\Users\\myusername\\test_data\\"
    # process_directory(input_dir)
    pass
    
# Example usage: whole direcory
# input_dir = "C:\\Users\\myusername\\test_data\\"
#process_directory(input_dir)
