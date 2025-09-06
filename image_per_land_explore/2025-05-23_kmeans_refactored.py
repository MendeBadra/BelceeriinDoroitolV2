# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from sklearn.cluster import KMeans
from scipy import ndimage
import imageio.v3 as iio
from pathlib import Path
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage import measure
import matplotlib.colors as mcolors

# %% Constants and Configurations
THRESHOLDS = {
    'worst': 0.1994333333,
    'bad': 0.1701666667,
    'medium': 0.1698666667,
    'good': 0.233,
    'best': 0.3486
}

# Define custom colormap for segmentation visualization (e.g., ground, grass, weed)
CUSTOM_COLORS_3_CLASS = ['#8B4513',  # brown (ground)
                         '#228B22',  # forest green (grass)
                         '#800080']  # purple (weed)
CUSTOM_CMAP_3_CLASS = ListedColormap(CUSTOM_COLORS_3_CLASS)

DEFAULT_KMEANS_LABELS = ["ground", "grass", "weed", "shrub", "bare", "shadow", "water", "other"]

IMAGE_SOURCES = {
    "worst": {
        "rgb_path": "2025-05-21_working_imgs/DJI_0020_124fplan.JPG",
        "index_path": "2025-05-21_working_imgs/DJI_002INDS.TIF"
    },
    "bad": {
        "rgb_path": "2025-05-21_working_imgs/DJI_0530_106fplan.JPG",
        "index_path": "2025-05-21_working_imgs/DJI_053INDS.TIF"
    },
    "medium": {
        "rgb_path": "medium/DJI_0010_oct.JPG",
        "index_path": "medium/DJI_001INDS.TIF"
    },
    "good": {
        "rgb_path": "good/DJI_0010_oct.JPG",
        "index_path": "good/DJI_001INDS.TIF"
    },
    "best": {
        "rgb_path": "best/DJI_0010oct.JPG",
        "index_path": "best/DJI_001INDS.TIF"
    }
}

# %% Utility Functions

def get_distinct_colormap(k):
    """Returns a ListedColormap with k visually distinct colors."""
    if k <= 10:
        base_cmap = plt.get_cmap('tab10')
        colors = base_cmap.colors[:k]
    elif k <= 20:
        base_cmap = plt.get_cmap('tab20')
        colors = base_cmap.colors[:k]
    else:
        base_cmap = plt.get_cmap('hsv')
        colors = [base_cmap(i / k) for i in range(k)]
    return mcolors.ListedColormap(colors)

def draw_masks(masks_dict, land_type="", figsize=(15, 5)):
    """
    Draws masks for different classes (e.g., weed, grass, ground).
    Assumes masks_dict contains 2D numpy arrays for 'weed', 'grass', 'ground'.
    """
    num_masks = len(masks_dict)
    if num_masks == 0:
        print("No masks to draw.")
        return

    fig, axes = plt.subplots(1, num_masks, figsize=figsize)
    if num_masks == 1: # Ensure axes is always iterable
        axes = [axes]

    for ax, (class_name, mask_image) in zip(axes, masks_dict.items()):
        ax.imshow(mask_image, cmap='gray')
        ax.set_title(class_name.capitalize())
        ax.axis('off')

    fig.suptitle(f"Masks for {land_type.upper()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

def draw_side_by_side(original_image, segmented_image, land_type="", k=3, cmap="viridis", figsize=(10, 5)):
    """Displays original (or processed) grayscale image and its K-means segmented version."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title(f"Input MSAVI (Masked) {land_type.upper()}")
    axs[0].axis('off')

    # Determine vmin and vmax from the segmented image for consistent display
    unique_cluster_values = np.unique(segmented_image)
    vmin = np.min(unique_cluster_values) if len(unique_cluster_values) > 0 else 0
    vmax = np.max(unique_cluster_values) if len(unique_cluster_values) > 0 else 1

    axs[1].imshow(segmented_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(f'K-means Segmented (k={k})')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    # return plt.gca() # Usually not needed if plt.show() is called

def segment_msavi(msavi_masked_input, k=3, class_labels=None):
    """
    Segments an MSAVI image using K-means clustering.
    Assumes higher intensity cluster centers correspond to denser vegetation.
    Labels (ground, grass, weed for k=3) are assigned based on sorted cluster center intensity.
    """
    h, w = msavi_masked_input.shape
    pixels = msavi_masked_input.reshape(-1, 1)

    # Filter out zero pixels for KMeans if they represent masked areas and are numerous
    # This can improve clustering if 0 is not a meaningful value for any class
    pixels_for_kmeans = pixels[pixels > 0].reshape(-1,1) # Consider only non-masked pixels
    if pixels_for_kmeans.shape[0] == 0: # All pixels are 0
        # Handle empty input: return empty/zero masks and an image of zeros
        print("Warning: Input to segment_msavi is all zeros. Returning zero masks.")
        segmented_image = np.zeros_like(msavi_masked_input)
        
        if class_labels is None:
            labels_to_use = DEFAULT_KMEANS_LABELS[:k]
        else:
            labels_to_use = class_labels[:k]
        
        empty_masks = {label: np.zeros_like(msavi_masked_input) for label in labels_to_use}
        return segmented_image, empty_masks, None # No KMeans model

    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(pixels_for_kmeans)
    
    # Create the full segmented image
    segmented_image = np.zeros((h * w), dtype=float)
    # Assign cluster centers to non-masked pixels
    segmented_image[pixels.flatten() > 0] = kmeans.cluster_centers_[kmeans.labels_].flatten()
    segmented_image = segmented_image.reshape(h, w)

    # Cluster centers determine class assignments (sorted by intensity)
    sorted_cluster_centers = sorted(kmeans.cluster_centers_.flatten())

    if len(sorted_cluster_centers) < k:
        # This might happen if k is too high for the data variance
        print(f"Warning: Only {len(sorted_cluster_centers)} unique clusters found, but k={k} requested. Adjusting k.")
        k_actual = len(sorted_cluster_centers)
    else:
        k_actual = k

    if class_labels is None:
        # Default labels: ground, grass, weed, etc.
        labels_to_use = DEFAULT_KMEANS_LABELS[:k_actual]
    else:
        labels_to_use = class_labels[:k_actual]
        if len(labels_to_use) < k_actual: # Ensure enough labels
            labels_to_use.extend(DEFAULT_KMEANS_LABELS[len(labels_to_use):k_actual])


    masks = {}
    for i in range(k_actual):
        label = labels_to_use[i]
        cluster_val = sorted_cluster_centers[i]
        # Create mask where segmented image equals the specific cluster center value
        # Important: Compare floating point numbers with a tolerance or use the labels directly
        # For simplicity here, direct comparison of cluster center values (results from KMeans)
        masks[label] = np.where(segmented_image == cluster_val, cluster_val, 0)
        # For binary masks: masks[label] = (segmented_image == cluster_val)

    return segmented_image, masks, kmeans

def clean_mask(binary_mask, closing_radius=1, opening_radius=6):
    """Cleans a binary mask using morphological closing and opening."""
    if np.sum(binary_mask) == 0: # Skip empty masks
        return binary_mask
    cleaned = binary_closing(binary_mask, disk(closing_radius))
    cleaned = binary_opening(cleaned, disk(opening_radius))
    return cleaned

def filter_small_objects_from_mask(binary_mask, min_size=1000, connectivity=2):
    """Labels and removes small objects from a binary mask."""
    if np.sum(binary_mask) == 0:
        return binary_mask
    label_image = measure.label(binary_mask, connectivity=connectivity)
    filtered_labels = remove_small_objects(label_image, min_size=min_size, connectivity=connectivity)
    return (filtered_labels > 0) # Return binary mask

def plot_labeled_image(image_data, subplot_position, title_prefix, cmap='nipy_spectral'):
    """Labels an image, displays it in a subplot, and sets a title."""
    labeled_array, num_features = ndimage.label(image_data)
    plt.subplot(*subplot_position)
    plt.imshow(labeled_array, cmap=cmap)
    plt.title(f"{title_prefix} labels ({num_features} features)")
    plt.axis('off')

def calculate_percentage(mask, land_type_name):
    """Calculates and prints the percentage of True pixels in a mask."""
    if mask.size == 0:
        print(f"[{land_type_name.upper()}] Mask is empty.")
        return 0.0
    truthy_pixel_count = np.sum(mask) # Assumes mask is boolean or 0/non-0
    total_pixels = mask.size
    percentage = (truthy_pixel_count / total_pixels) * 100
    print(f"[{land_type_name.upper()}] Mask coverage: {percentage:.2f}%")
    return percentage

def analyze_region(label_index, labeled_array, original_binary_mask, msavi_data=None, plot=False):
    """
    Analyzes a specific labeled region from a morphologically cleaned and filtered mask,
    referring back to the original (KMeans output) binary mask for sparsity.
    """
    if not isinstance(original_binary_mask, np.ndarray) or original_binary_mask.dtype != bool:
        raise ValueError(f"original_binary_mask must be a boolean numpy array. Got type: {original_binary_mask.dtype}")

    object_slices = ndimage.find_objects(labeled_array)
    if not (1 <= label_index <= len(object_slices)) or object_slices[label_index - 1] is None:
        print(f"Label index {label_index} is out of bounds or refers to a removed object.")
        return 0, 0.0

    slc = object_slices[label_index - 1]
    # region_mask_in_slice is the specific labeled object within its bounding box
    region_mask_in_slice = (labeled_array[slc] == label_index)
    
    area_of_labeled_object = np.sum(region_mask_in_slice)
    if area_of_labeled_object == 0:
        return 0, 0.0

    # Extract corresponding part from original (KMeans) binary weed mask
    original_weed_pixels_in_slice = original_binary_mask[slc]
    
    # Intersect the labeled object's shape with the original weed pixels within that shape
    weed_pixels_within_labeled_object = original_weed_pixels_in_slice[region_mask_in_slice]
    
    num_actual_weed_pixels_in_object = np.sum(weed_pixels_within_labeled_object)
    
    # Sparsity/Coverage: Percentage of the labeled object's area that was 'weed' in the original Kmeans mask
    coverage_percentage = (num_actual_weed_pixels_in_object / area_of_labeled_object) * 100

    if plot:
        print(f"--- Analysis for Label {label_index} ---")
        print(f"Bounding box slice: {slc}")
        print(f"Area of morphologically cleaned object: {area_of_labeled_object} pixels")
        print(f"Original weed coverage within this object: {coverage_percentage:.2f}%")

        fig_cols = 3 if msavi_data is not None else 2
        plt.figure(figsize=(5 * fig_cols, 5))

        plt.subplot(1, fig_cols, 1)
        plt.imshow(labeled_array[slc] == label_index, cmap='gray') # Show specific object
        plt.title(f"Cleaned Object (Label {label_index})")
        plt.axis('off')

        plt.subplot(1, fig_cols, 2)
        # Show original KMeans weed pixels within the bounds of this cleaned object
        display_original_weed_region = np.zeros_like(region_mask_in_slice, dtype=bool)
        display_original_weed_region[region_mask_in_slice] = weed_pixels_within_labeled_object
        plt.imshow(display_original_weed_region, cmap='gray')
        plt.title("Original K-Means 'Weed' \nwithin Cleaned Object")
        plt.axis('off')

        if msavi_data is not None:
            msavi_slice = msavi_data[slc]
            msavi_region_display = np.full(msavi_slice.shape, np.nan) # Mask with NaN
            # Show MSAVI values only where the cleaned object is
            msavi_region_display[region_mask_in_slice] = msavi_slice[region_mask_in_slice]

            plt.subplot(1, fig_cols, 3)
            plt.imshow(msavi_region_display, cmap='viridis', vmin=0, vmax=1) # Assuming MSAVI range
            plt.colorbar(label="MSAVI")
            plt.title("MSAVI within Cleaned Object")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return area_of_labeled_object, coverage_percentage

# %% Core Processing Function
def process_land_patch(name, msavi_raw, ground_threshold, k=3, segmentation_cmap=CUSTOM_CMAP_3_CLASS, class_labels=None):
    """Processes a single land patch: masks, segments, and visualizes."""
    msavi_no_nan = np.nan_to_num(msavi_raw, nan=0.0)

    # Create ground mask: pixels below threshold are considered ground (masked out)
    ground_mask_binary = msavi_no_nan < ground_threshold
    msavi_veg_only = msavi_no_nan * ~ground_mask_binary # Apply mask, keeping only vegetation

    segmented_image, masks, kmeans_model = segment_msavi(msavi_veg_only, k=k, class_labels=class_labels)

    if kmeans_model: # Check if KMeans ran successfully
        print(f"[{name.upper()}] Cluster centers: {sorted(kmeans_model.cluster_centers_.flatten())}")
    else:
        print(f"[{name.upper()}] KMeans did not run (likely empty input after masking).")

    draw_side_by_side(msavi_veg_only, segmented_image, land_type=name, cmap=segmentation_cmap, k=k)
    
    # Select only ground, grass, weed for draw_masks if k=3 and default labels used
    if k == 3 and (class_labels is None or class_labels[:3] == ["ground", "grass", "weed"]):
        display_masks = {
            "ground": masks.get("ground", np.zeros_like(msavi_raw)),
            "grass": masks.get("grass", np.zeros_like(msavi_raw)),
            "weed": masks.get("weed", np.zeros_like(msavi_raw))
        }
        draw_masks(display_masks, land_type=name)
    else: # Draw all available masks
        draw_masks(masks, land_type=name)


    return {
        "msavi_veg_only": msavi_veg_only,
        "segmented_image": segmented_image,
        "masks": masks, # These masks contain cluster center values, not binary
        "kmeans_model": kmeans_model,
        "ground_mask_binary": ground_mask_binary
    }

# %% Main Execution / Example Usage (Original Script Flow)
if __name__ == "__main__":
    # 1. Load Raw Inputs
    raw_inputs = {}
    for name, paths in IMAGE_SOURCES.items():
        try:
            # Assuming RGB is not directly used in this part of the workflow for processing
            # rgb = iio.imread(paths["rgb_path"])
            indices = iio.imread(paths["index_path"])
            msavi = indices[:, :, 0] # Assuming MSAVI is the first band
            raw_inputs[name] = {"msavi": msavi}
        except FileNotFoundError:
            print(f"Error: File not found for {name}. Path: {paths['index_path']} or {paths['rgb_path']}")
        except Exception as e:
            print(f"Error loading data for {name}: {e}")


    # 2. Process each land type
    results = {}
    for name, data in raw_inputs.items():
        print(f"\n--- Processing Land Type: {name.upper()} ---")
        if 'msavi' not in data:
            continue
        
        patch_threshold = THRESHOLDS.get(name)
        if patch_threshold is None:
            print(f"Warning: No threshold found for {name}. Skipping.")
            continue
        
        # For k=3, using default ground, grass, weed interpretation
        results[name] = process_land_patch(name, data["msavi"],
                                           ground_threshold=patch_threshold,
                                           k=3,
                                           segmentation_cmap=CUSTOM_CMAP_3_CLASS,
                                           class_labels=["ground", "grass", "weed"]) # Explicitly pass labels

    # 3. Example: Further analysis on 'best' and 'bad' types (as in original script)
    if 'best' in results and 'bad' in results:
        print("\n--- Detailed Weed Mask Analysis ---")

        # Convert Kmeans output masks to binary for morphology (True for any non-zero pixel)
        best_weed_mask_kmeans_raw = results['best']['masks'].get('weed', np.zeros_like(raw_inputs['best']['msavi'])) != 0
        bad_weed_mask_kmeans_raw = results['bad']['masks'].get('weed', np.zeros_like(raw_inputs['bad']['msavi'])) != 0

        # Clean these binary KMeans weed masks
        # Parameters for cleaning can be tuned
        rad_close, rad_open = 1, 6
        best_weed_cleaned = clean_mask(best_weed_mask_kmeans_raw, rad_close, rad_open)
        bad_weed_cleaned = clean_mask(bad_weed_mask_kmeans_raw, rad_close, rad_open)
        
        # Filter small objects from cleaned masks
        min_object_size = 5000
        best_weed_filtered = filter_small_objects_from_mask(best_weed_cleaned, min_size=min_object_size)
        bad_weed_filtered = filter_small_objects_from_mask(bad_weed_cleaned, min_size=min_object_size)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(bad_weed_filtered, cmap="gray")
        plt.title("Bad - Filtered Weed Mask")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(best_weed_filtered, cmap="gray")
        plt.title("Best - Filtered Weed Mask")
        plt.axis('off')
        plt.suptitle("Morphologically Cleaned and Filtered Weed Masks", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Calculate percentages
        print("\n--- Weed Coverage Percentages (after cleaning & filtering) ---")
        calculate_percentage(best_weed_filtered, "Best (filtered weed)")
        calculate_percentage(bad_weed_filtered, "Bad (filtered weed)")
        
        print("\n--- Weed Coverage Percentages (raw K-Means output) ---")
        calculate_percentage(best_weed_mask_kmeans_raw, "Best (raw K-Means weed)")
        calculate_percentage(bad_weed_mask_kmeans_raw, "Bad (raw K-Means weed)")

        # Label connected regions in the filtered masks
        best_labeled_filtered, best_num_features = ndimage.label(best_weed_filtered)
        bad_labeled_filtered, bad_num_features = ndimage.label(bad_weed_filtered)

        print(f"\n'Best' type has {best_num_features} distinct large weed regions after filtering.")
        print(f"'Bad' type has {bad_num_features} distinct large weed regions after filtering.")

        plt.figure(figsize=(12, 6))
        plot_labeled_image(best_labeled_filtered, (1, 2, 1), "Best (Filtered)")
        plot_labeled_image(bad_labeled_filtered, (1, 2, 2), "Bad (Filtered)")
        plt.suptitle("Labeled Weed Regions (Post-Cleaning & Filtering)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        # Analyze some regions (example)
        if best_num_features > 0:
            print("\n--- Analyzing sample region from 'BEST' data ---")
            # Ensure msavi data is available
            best_msavi_data = raw_inputs.get('best', {}).get('msavi')
            if best_msavi_data is not None:
                 # analyze_region expects original Kmeans mask to be boolean
                analyze_region(label_index=1,  # Analyze the first largest object
                               labeled_array=best_labeled_filtered,
                               original_binary_mask=best_weed_mask_kmeans_raw, # Original KMeans output (binary)
                               msavi_data=best_msavi_data,
                               plot=True)
            else:
                print("MSAVI data for 'best' not available for region analysis.")

        if bad_num_features > 0:
            print("\n--- Analyzing sample region from 'BAD' data ---")
            bad_msavi_data = raw_inputs.get('bad', {}).get('msavi')
            if bad_msavi_data is not None:
                analyze_region(label_index=1,
                               labeled_array=bad_labeled_filtered,
                               original_binary_mask=bad_weed_mask_kmeans_raw, # Original KMeans output (binary)
                               msavi_data=bad_msavi_data,
                               plot=True)
            else:
                print("MSAVI data for 'bad' not available for region analysis.")

    # Example for k=4 segmentation
    # print("\n--- Example: K=4 Segmentation for 'BEST' ---")
    # if 'best' in raw_inputs and 'msavi' in raw_inputs['best']:
    #     k_custom = 4
    #     custom_cmap_k4 = get_distinct_colormap(k_custom)
    #     process_land_patch(name="best_k4",
    #                        msavi_raw=raw_inputs['best']['msavi'],
    #                        ground_threshold=THRESHOLDS['best'],
    #                        k=k_custom,
    #                        segmentation_cmap=custom_cmap_k4) # No specific labels for k=4 by default