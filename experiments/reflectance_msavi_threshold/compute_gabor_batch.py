"""This module computes the gabor algorithm from the aligned dir which is currently deleted due to space constraints
INPUT: aligned_root_dir
This is the gabor segmentation and thresholding part of the whole workflow. 
"""
import numpy as np
import rasterio
from skimage import color, exposure # Removed img_as_ubyte unless you confirm its necessity and correct usage
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for matplotlib when saving figures in scripts
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc # Garbage Collector interface


from gabor.plot_utils import gabor_segmentation_mask


THRESHOLDS = {
    'worst': 0.1994333333,
    'bad': 0.1701666667,
    'medium': 0.1698666667,
    'good': 0.233,
    'best': 0.3486
}

PLOT_SAVE_DIR = Path('drone_data_output_plots_2025-05-14')

TEST_THRESHOLD_OFFSET = 0.2

ALIGNED_ROOT_DIR = Path("drone_data_2024_october_aligned")


# Deprecated in favor of stretching to percentile values
# def convert_to_uint8(img_uint16):
#     # Normalize to 0–255 range, then convert
#     assert img_uint16.dtype == np.uint16, "Input image must be of type uint16"
#     img_norm = (img_uint16 / img_uint16.max()) * 255
#     return img_norm.astype(np.uint8)
# --- End of Placeholders ---
def to_reflectance_float32(image_uint16):
    """
    Converts a uint16 image array to a float32 array with values scaled to [0, 1]
    representing reflectance. The scaling is done by dividing by the maximum possible
    uint16 value (2**16 - 1).

    Args:
        image_uint16 (np.ndarray): Input image array with dtype uint16.

    Returns:
        np.ndarray: Output image array with dtype float32 and values in [0, 1].
    """
    assert image_uint16.dtype == np.uint16, "Input image must be of type uint16"
    return image_uint16.astype(np.float32) / (2**16 - 1)    

def multiply_band(band, factor):
    return band * factor

def stretch_band(band, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.percentile(band, 2)
    if max_val is None:
        max_val = np.percentile(band, 98)
    stretched = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    return stretched

def compute_gabor_mask(rgb, msavi, threshold, experimental=False, save_plots=False, plot_save_path=None):
    """
    Computes Gabor mask and related metrics.
    Saves plots directly if save_plots is True and plot_save_path is provided.
    Does NOT return the plot figure object to save memory.
    """
    if rgb is None or msavi is None:
        # Handle cases where essential data might be missing
        error_key_prefix = "test_" if experimental else ""
        return {
            f"{error_key_prefix}threshold": threshold,
            f"{error_key_prefix}weed_percentage": None,
            f"{error_key_prefix}healthy_vegetation_percentage": None,
            f"{error_key_prefix}overlap_weed_ground_percentage": None,
            f"{error_key_prefix}high_intensity_section_percentage": None, # if experimental
            "error": "Missing RGB or MSAVI data for Gabor"
        }

    vegetation_mask = msavi > threshold
    gray = color.rgb2gray(rgb)

    soil_removed_gray = gray * vegetation_mask
    # Adaptive histogram equalization can be memory intensive on very large images.
    # Ensure soil_removed_gray is not excessively large or consider alternatives if this is a bottleneck.
    local_hist_equalized = exposure.equalize_adapthist(soil_removed_gray.astype(np.float32), clip_limit=0.03) # Ensure float input

    # Assuming gabor_segmentation_mask handles memory of its internals.
    # binary_mask_objects might hold large data.
    binary_mask_objects, _ = gabor_segmentation_mask([soil_removed_gray, local_hist_equalized], frequencies=[0.5, 2.5], angles=[0, np.pi/2])
    weed_mask = binary_mask_objects[0].get() # This should be a 2D boolean array

    # Explicitly delete intermediate arrays to free memory sooner
    del gray, soil_removed_gray, local_hist_equalized, binary_mask_objects
    gc.collect() # Suggest garbage collection

    ground_mask = ~vegetation_mask # Inverted vegetation mask
    overlap_weed_ground = np.logical_and(weed_mask, ground_mask)

    # Calculate percentages
    total_pixels_in_mask_domain = weed_mask.size # Should be same as msavi.size or vegetation_mask.size
    if total_pixels_in_mask_domain == 0: # Avoid division by zero
        weed_percentage = 0.0
        healthy_vegetation_percentage = 0.0
        overlap_weed_ground_percentage = 0.0
        current_veg_pixels = 0
    else:
        weed_pixels = weed_mask.sum()
        current_veg_pixels = vegetation_mask.sum()

        weed_percentage = round(weed_pixels / total_pixels_in_mask_domain * 100, 4)
        
        # Healthy vegetation: Pixels that are in vegetation_mask AND NOT in weed_mask
        healthy_veg_pixels = np.logical_and(vegetation_mask, ~weed_mask).sum()
        healthy_vegetation_percentage = round(healthy_veg_pixels / total_pixels_in_mask_domain * 100, 4)
        
        overlap_weed_ground_percentage = round(overlap_weed_ground.sum() / total_pixels_in_mask_domain * 100, 4)


    if save_plots and plot_save_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Image")
        axes[0].axis('off')

        veg_percentage_display = current_veg_pixels / total_pixels_in_mask_domain if total_pixels_in_mask_domain > 0 else 0
        
        # For the display of "soil removed gray", use the original rgb data masked
        display_soil_removed = rgb * vegetation_mask[..., np.newaxis]
        axes[1].imshow(display_soil_removed) # Show color version
        axes[1].set_title(f"Veg Masked RGB (veg={veg_percentage_display:.2f})")
        axes[1].axis('off')
        del display_soil_removed

        axes[2].imshow(weed_mask, cmap='binary')
        axes[2].set_title("Gabor Weed Mask" if not experimental else "[TEST] Gabor Weed Mask")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(plot_save_path)
        plt.close(fig) # CRITICAL: Close the figure to free memory
        del fig, axes # Further ensure figure objects are released
        gc.collect()

    # Prepare results dictionary
    update_dict = {
        "threshold": threshold, # This is the threshold used for this specific gabor run
        "weed_percentage": weed_percentage,
        "healthy_vegetation_percentage": healthy_vegetation_percentage,
        "overlap_weed_ground_percentage": overlap_weed_ground_percentage,
    }

    if experimental:
        # "high_intensity_section_percentage" refers to vegetation percent at THIS experimental threshold
        high_intensity_percentage = round(current_veg_pixels / total_pixels_in_mask_domain * 100, 4) if total_pixels_in_mask_domain > 0 else 0
        test_result_dict = {
            "test_threshold": threshold,
            "test_weed_percentage": update_dict['weed_percentage'],
            "test_healthy_vegetation_percentage": update_dict['healthy_vegetation_percentage'], # Added
            "test_overlap_weed_ground_percentage": update_dict['overlap_weed_ground_percentage'],
            "test_high_intensity_section_percentage": high_intensity_percentage,
        }
        del vegetation_mask, weed_mask, ground_mask, overlap_weed_ground
        gc.collect()
        return test_result_dict
    else:
        del vegetation_mask, weed_mask, ground_mask, overlap_weed_ground
        gc.collect()
        return update_dict


def process_all(drone_image_number, land_type, base_threshold, fplan_dir: Path, save_plots=True):
    result = {
        'drone_image_number': drone_image_number,
        'land_type': land_type,
        'fplan': fplan_dir.name,
        'base_threshold_for_veg_percent': base_threshold # Clarify this is the base for initial veg %
    }
    
    msavi = None
    rgb = None # Initialize rgb to None

    # Create save directory for plots if needed
    save_dir = PLOT_SAVE_DIR / land_type / fplan_dir.name
    if save_plots:
        save_dir.mkdir(parents=True, exist_ok=True)

    inds_path = fplan_dir / f"DJI_{drone_image_number}INDS.TIF"
    if inds_path.exists():
        try:
            with rasterio.open(inds_path) as src:
                msavi = src.read(1).astype(np.float32)
            # Calculate initial vegetation stats based on the base_threshold
            initial_vegetation_mask = msavi > base_threshold
            total_pixels = msavi.size
            initial_vegetation_percent = initial_vegetation_mask.sum() / total_pixels * 100 if total_pixels > 0 else 0
            result.update({
                'initial_vegetation_percent': round(initial_vegetation_percent, 4),
                'initial_ground_percent': round(100 - initial_vegetation_percent, 4)
            })
            del initial_vegetation_mask # Free memory
        except Exception as e:
            print(f"Error reading MSAVI {inds_path}: {e}")
            msavi = None # Ensure msavi is None if reading failed
            result.update({'initial_vegetation_percent': None, 'initial_ground_percent': None, 'msavi_error': str(e)})
    else:
        print(f"MSAVI file not found: {inds_path}")
        result.update({'initial_vegetation_percent': None, 'initial_ground_percent': None, 'msavi_error': 'File not found'})

    bands_path = fplan_dir / f"DJI_{drone_image_number}BANDS.TIF"
    if bands_path.exists():
        try:
            with rasterio.open(bands_path) as src:
                blue = src.read(1).astype(np.uint16)
                green = src.read(2).astype(np.uint16)
                red = src.read(3).astype(np.uint16)
                blue_float = to_reflectance_float32(blue)
                green_float = to_reflectance_float32(green)
                red_float = to_reflectance_float32(red)
                # TODO: Fix this
                rgb_reflectance = np.stack((red_float, green_float, blue_float), axis=-1) # Common RGB order
                rgb = stretch_band(rgb_reflectance) # Stretch to [0, 1] range
                # rgb = (rgb_reflectance * 10).copy()
                # rgb = multiply_band(rgb_reflectance, 10) # Scale to [0, 10] range
                rgb = rgb_reflectance * (1/np.max(rgb_reflectance)) # Normalize to [0, 1]
            del blue, green, red # Free memory of individual bands
        except Exception as e:
            print(f"Error reading BANDS {bands_path}: {e}")
            rgb = None # Ensure rgb is None if reading failed
            result.update({'bands_error': str(e)})
    else:
        print(f"BANDS file not found: {bands_path}")
        result.update({'bands_error': 'File not found'})

    # Proceed with Gabor if msavi and rgb were loaded successfully
    if msavi is not None and rgb is not None:
        # --- Main Gabor processing ---
        plot_main_path = save_dir / f"{land_type}_{fplan_dir.name}_{drone_image_number}_gabor_main.jpg" if save_plots else None
        main_gabor_results = compute_gabor_mask(rgb, msavi, base_threshold, 
                                                experimental=False, save_plots=save_plots, 
                                                plot_save_path=plot_main_path)
        result.update(main_gabor_results)

        # Validation check (optional, but good for debugging)
        # This 'healthy_vegetation_percentage' is from gabor, calculated as (veg_mask & ~weed_mask).sum() / total_size
        # initial_vegetation_percent is (veg_mask).sum() / total_size
        # weed_percentage from gabor is weed_mask.sum() / total_size
        # So, initial_veg_percent - weed_percent is not necessarily healthy_veg_percent if weed can be outside initial veg.
        # The definition in compute_gabor_mask is more robust for "healthy vegetation pixels".
        if 'initial_vegetation_percent' in result and result['initial_vegetation_percent'] is not None and \
           'weed_percentage' in main_gabor_results and main_gabor_results['weed_percentage'] is not None and \
           'healthy_vegetation_percentage' in main_gabor_results and main_gabor_results['healthy_vegetation_percentage'] is not None:

            # This is the vegetation percentage at the base_threshold (same as initial_vegetation_percent)
            veg_at_base_thresh_pct = round((msavi > base_threshold).sum() / msavi.size * 100, 4)
            
            # Recalculate healthy based on veg at base_threshold MINUS weed at base_threshold.
            # This assumes weed is a subset of total area, not just initial vegetation area.
            calc_healthy_pct = veg_at_base_thresh_pct - main_gabor_results['weed_percentage']

            if not np.isclose(main_gabor_results['healthy_vegetation_percentage'], calc_healthy_pct, atol=0.1): # Increased tolerance
                 # print(f"Warning: Healthy Veg % mismatch for {drone_image_number} (main). "
                 #      f"Gabor_HVP: {main_gabor_results['healthy_vegetation_percentage']:.2f}, "
                 #      f"Veg({base_threshold:.2f}) - Gabor_WP: {calc_healthy_pct:.2f}, "
                 #      f"Veg({base_threshold:.2f}): {veg_at_base_thresh_pct:.2f}, "
                 #      f"Gabor_WP: {main_gabor_results['weed_percentage']:.2f}")
                 result["healthy_vegetation_percentage_consistency_check"] = round(calc_healthy_pct, 4)


        # --- Experimental Gabor processing ---
        experimental_threshold = base_threshold + TEST_THRESHOLD_OFFSET
        plot_test_path = save_dir / f"{land_type}_{fplan_dir.name}_{drone_image_number}_gabor_test.jpg" if save_plots else None
        test_gabor_results = compute_gabor_mask(rgb, msavi, experimental_threshold, 
                                                experimental=True, save_plots=save_plots, 
                                                plot_save_path=plot_test_path)
        result.update(test_gabor_results)
        
        del rgb # rgb is no longer needed

    elif msavi is None or rgb is None: # If data loading failed, fill with None
        keys_to_nullify = [
            "threshold", "weed_percentage", "healthy_vegetation_percentage", "overlap_weed_ground_percentage",
            "test_threshold", "test_weed_percentage", "test_healthy_vegetation_percentage",
            "test_overlap_weed_ground_percentage", "test_high_intensity_section_percentage"
        ]
        for key in keys_to_nullify:
            if key not in result: # Don't overwrite error messages if they were set
                 result[key] = None
    
    del msavi # msavi is no longer needed
    gc.collect() # Explicitly call garbage collector at the end of a heavy task
    return result


def process_all_task(task_args):
    # Unpack arguments for process_all
    drone_image_number, land_type, threshold, fplan_dir, save_plots_flag = task_args
    try:
        return process_all(drone_image_number, land_type, threshold, fplan_dir, save_plots_flag)
    except Exception as e:
        print(f"Critical error in process_all_task for {drone_image_number} in {fplan_dir.name}: {e}")
        # Return a dictionary with error information to maintain DataFrame structure
        return {
            'drone_image_number': drone_image_number,
            'land_type': land_type,
            'fplan': fplan_dir.name,
            'base_threshold_for_veg_percent': threshold,
            'critical_error': str(e)
        }

def test():
    """
    Test function to demonstrate the processing call.
    """
    print("Starting test function...")

    # these are the inputs
    drone_image_number = "027"
    # input_dir = Path("test/TESTFPLAN") # This was commented out in your example

    # Ensure the test directory exists for the Path object, or handle appropriately
    # For this example, we'll just define it. If process_all needs it to exist,
    # you might want to create it in a setup step or within process_all.
    input_aligned_dir_path_str = "test/TESTFPLAN_aligned"
    input_aligned_dir = Path(input_aligned_dir_path_str)

    # Optional: Create the directory if it doesn't exist to avoid errors in process_all
    # input_aligned_dir.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured directory exists: {input_aligned_dir.resolve()}")


    quality_level_to_test = "bad"
    selected_threshold = THRESHOLDS.get(quality_level_to_test)

    if selected_threshold is None:
        print(f"Error: Threshold for quality level '{quality_level_to_test}' not found in THRESHOLDS.")
        return # Or raise an error

    print(f"Processing drone image: {drone_image_number}")
    print(f"Quality level: {quality_level_to_test}")
    print(f"Using threshold: {selected_threshold}")
    print(f"Input aligned directory: {input_aligned_dir}")

    result = process_all(
        drone_image_number,
        quality_level_to_test,
        selected_threshold,
        input_aligned_dir,
        save_plots=True
    )

    print("Test function completed.")
    print(result)
# --- Main execution logic ---
if __name__ == "__main__":
    # Create dummy files and directories for testing if they don't exist
    (PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)


    tasks = []
    for category_dir in Path(ALIGNED_ROOT_DIR).iterdir():
        if category_dir.is_dir():
            land_type_from_folder = category_dir.name.lower().split('_')[0]
            # Use a default threshold if specific one not found, or skip
            threshold = THRESHOLDS.get(land_type_from_folder, THRESHOLDS.get("default")) 
            if threshold is None:
                print(f"Skipping {category_dir.name}: No threshold found for land_type '{land_type_from_folder}' and no default.")
                continue
            
            for fplan_dir in category_dir.iterdir():
                if fplan_dir.is_dir():
                    for inds_path in fplan_dir.glob("DJI_*INDS.TIF"):
                        drone_image_number = inds_path.stem.replace("DJI_", "").replace("INDS", "")
                        # Add task: (drone_image_number, land_type, threshold, fplan_dir, save_plots_flag)
                        tasks.append((drone_image_number, land_type_from_folder, threshold, fplan_dir, True)) 
    
    print(f"Generated {len(tasks)} tasks.")

    if not tasks:
        print("No tasks to process. Check ALIGNED_ROOT_DIR, THRESHOLDS, and file naming conventions.")
    else:
        # IMPORTANT: Adjust max_workers based on your system's memory and CPU cores.
        # Start with a small number (e.g., 2 or 4) and monitor memory usage.
        # If each task uses 2GB, and you have 16GB RAM, you might try 4-6 workers.
        num_workers = 10 # conservative default, ADJUST THIS! 
        print(f"Running tasks with up to {num_workers} workers...")
        
        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Using executor.map for ordered results, wrapped with tqdm for progress
            all_results = list(tqdm(executor.map(process_all_task, tasks), total=len(tasks)))

        # Filter out potential None results if any critical error happened before dict creation
        results_df = pd.DataFrame([res for res in all_results if isinstance(res, dict)])
        
        print("\nSample of results:")
        print(results_df.head())
        
        output_csv_path = PLOT_SAVE_DIR / "processing_summary_results.csv"
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete. Results saved to {output_csv_path}")

        # Basic error summary (if 'critical_error' or other error columns exist)
        if 'critical_error' in results_df.columns:
            errors_df = results_df[results_df['critical_error'].notna()]
            if not errors_df.empty:
                print(f"\nEncountered {len(errors_df)} critical errors during processing.")
                print(errors_df[['drone_image_number', 'fplan', 'critical_error']].head())

        if 'msavi_error' in results_df.columns:
             msavi_errors_df = results_df[results_df['msavi_error'].notna()]
             if not msavi_errors_df.empty:
                print(f"\nEncountered {len(msavi_errors_df)} MSAVI loading errors.")

        if 'bands_error' in results_df.columns:
             bands_errors_df = results_df[results_df['bands_error'].notna()]
             if not bands_errors_df.empty:
                print(f"\nEncountered {len(bands_errors_df)} BANDS loading errors.")