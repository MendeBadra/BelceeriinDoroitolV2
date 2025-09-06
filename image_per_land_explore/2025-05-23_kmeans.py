# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.exposure import rescale_intensity
from sklearn.cluster import KMeans
from scipy import ndimage
from matplotlib.colors import ListedColormap

import marimo as mo

import imageio.v3 as iio
from pathlib import Path

THRESHOLDS = {
    'worst': 0.1994333333,
    'bad': 0.1701666667,
    'medium': 0.1698666667,
    'good': 0.233,
    'best': 0.3486
}



# %%
custom_colors = ['#8B4513',  # brown
                 '#228B22',  # forest green
                 '#800080']  # purple

# Create the custom colormap
custom_cmap = ListedColormap(custom_colors)

# %%
def draw_masks(weed, grass, ground, land_type=""):
    # Assuming weed, grass, and ground are 2D arrays (same shape)
    fig, ax = plt.subplots(1, 3, figsize=(150, 50))  # 1 row, 3 columns
    
    ax[0].imshow(weed, cmap='gray')
    ax[0].set_title('Weed')
    ax[0].axis('off')
    
    ax[1].imshow(grass, cmap='gray')
    ax[1].set_title('Grass')
    ax[1].axis('off')
    
    ax[2].imshow(ground, cmap='gray')
    ax[2].set_title('Ground')
    ax[2].axis('off')
    
    fig.suptitle(land_type.upper())
    plt.tight_layout()
    plt.show()
def draw_side_by_side(msavi_masked, segmented_image, land_type="", k=3, cmap="viridis"):
    # Display original and segmented image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(msavi_masked, cmap='gray')
    axs[0].set_title(f"Original Grayscale Image {land_type.upper()}")
    axs[0].axis('off')
    

    # Change ---
    vmin = np.min(segmented_image)
    vmax = np.max(segmented_image)
    segmented_image_normalized = (segmented_image - vmin) / (vmax - vmin)
    # ----
    axs[1].imshow(segmented_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(f'K-means Segmented (k={k})')
    axs[1].axis('off')
    
    plt.tight_layout()
    return plt.gca()


# def segment_msavi(msavi_masked, k=3):
#     h, w = msavi_masked.shape
#     pixels = msavi_masked.reshape(-1, 1)
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    
#     segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
#     segmented_image = segmented_pixels.reshape(h, w)
    
#     # Classify into ground, grass, weed based on sorted brightness
#     sorted_vals = sorted(np.unique(segmented_pixels))
#     if len(sorted_vals) < 3:
#         raise ValueError("Not enough clusters detected to assign all 3 classes.")
    
#     ground_val, grass_val, weed_val = sorted_vals
    
#     masks = {
#         "ground": np.where(segmented_image == ground_val, segmented_image, 0),
#         "grass":  np.where(segmented_image == grass_val, segmented_image, 0),
#         "weed":   np.where(segmented_image == weed_val, segmented_image, 0),
#     }

#     return segmented_image, masks, kmeans

def segment_msavi(msavi_masked, k=3):
    h, w = msavi_masked.shape
    pixels = msavi_masked.reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_pixels.reshape(h, w)

    unique_vals = sorted(np.unique(segmented_pixels))
    if len(unique_vals) < k:
        raise ValueError(f"Only {len(unique_vals)} unique clusters found, but k={k} requested.")

    # Generate class names dynamically
    default_labels = ["ground", "grass", "weed", "shrub", "bare", "shadow", "other"]
    labels = default_labels[:k]

    masks = {
        label: np.where(segmented_image == val, segmented_image, 0)
        for label, val in zip(labels, unique_vals)
    }

    return segmented_image, masks, kmeans


# %%
def process_land_patch(name, msavi_raw, ground_mask=None, k=3, cmap="viridis"):
    msavi_nonnan = np.nan_to_num(msavi_raw, nan=0.0)
    msavi_masked = msavi_nonnan if ground_mask is None else msavi_nonnan * ~ground_mask
    
    segmented_image, masks, kmeans = segment_msavi(msavi_masked, k)

    print(f"[{name}] Cluster centers: {sorted(kmeans.cluster_centers_.flatten())}")
    draw_side_by_side(msavi_masked, segmented_image, land_type=name, cmap=cmap, k=k)
    draw_masks(masks["weed"], masks["grass"], masks["ground"], land_type=name)

    return {
        "masked": msavi_masked,
        "segmented": segmented_image,
        "masks": masks,
        "kmeans": kmeans
    }


# %%
image_sources = {
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


# # Inputs:
# worst_rgb = iio.imread("2025-05-21_working_imgs/DJI_0020_124fplan.JPG")
# bad_rgb = iio.imread("2025-05-21_working_imgs/DJI_0530_106fplan.JPG")

# worst_indices = iio.imread("2025-05-21_working_imgs/DJI_002INDS.TIF")
# bad_indices = iio.imread("2025-05-21_working_imgs/DJI_053INDS.TIF")
# worst_msavi = worst_indices[:,:,0]
# bad_msavi = bad_indices[:,:,0]

# THRESHOLDS = {
#     'worst': 0.1994333333,
#     'bad': 0.1701666667,
#     'medium': 0.1698666667,
#     'good': 0.233,
#     'best': 0.3486
# }


# %%
raw_inputs = {}

for name, paths in image_sources.items():
    rgb = iio.imread(paths["rgb_path"])
    indices = iio.imread(paths["index_path"])
    msavi = indices[:, :, 0]

    raw_inputs[name] = {
        "rgb": rgb,
        # "indices": indices,
        "msavi": msavi,
    }


# %%
# Process
GROUND_MASKS = {}



for key, data in raw_inputs.items():
    ground_mask = data['msavi'] < THRESHOLDS[key]
    GROUND_MASKS[key] = ground_mask

# %%
results = {}

for name, data in raw_inputs.items():
    mask = GROUND_MASKS.get(name, None)
    results[name] = process_land_patch(name, data["msavi"], ground_mask=mask, cmap=custom_cmap)

# %%
results

# %%
# from matplotlib.colors import LinearSegmentedColormap

# # Define your color gradient stops: ground → grass → weed
# custom_colors = ['#8B4513',  # brown (ground)
#                  '#228B22',  # forest green (grass)
#                  '#800080']  # purple (weed)

# # Create a smooth colormap with linear interpolation between the colors
# custom_cmap = LinearSegmentedColormap.from_list("smooth_vegmap", custom_colors, N=256)

# %%
plt.imshow(results['best']['segmented'], cmap=custom_cmap)

# %%
plt.imshow(results['best']['masks']['weed'], cmap=ListedColormap(['#000000','#228B22']))

# %%
import matplotlib.colors as mcolors
def get_distinct_colormap(k):
    """
    Returns a ListedColormap with k visually distinct colors.
    """
    # Try tab20 or tab10 first (good for up to 20 classes)
    if k <= 20:
        base = plt.get_cmap('tab20' if k > 10 else 'tab10')
        colors = base.colors[:k]
    else:
        # For k > 20, fallback to hsv or viridis sampled at regular intervals
        base = plt.get_cmap('hsv')
        colors = [base(i / k) for i in range(k)]

    return mcolors.ListedColormap(colors)

k = 4
disting_cmap = get_distinct_colormap(k)

# %%
results_k_4_means = {}


for name, data in raw_inputs.items():
    mask = GROUND_MASKS.get(name, None)
    results_k_4_means[name] = process_land_patch(name, data["msavi"], ground_mask=mask, k=4, cmap=disting_cmap)

# %% [markdown]
# 2025.05.26

# %%
plt.imshow(results['worst']['masks']['weed'], cmap="gray")


# %%
bad_weed = results['bad']['masks']['weed']
best_weed = results['best']['masks']['weed']
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(bad_weed, cmap="gray")
plt.title("Bad weed")
plt.subplot(1, 2, 2)
plt.imshow(best_weed, cmap="gray")
plt.title("Best weed")

# %%
from skimage.morphology import binary_closing, binary_opening, disk

def clean_mask(weed_mask, rad1=1, rad2=6):
    weed_mask_clean = binary_closing(weed_mask, disk(rad1))
    weed_mask_clean = binary_opening(weed_mask_clean, disk(rad2))
    return weed_mask_clean


rad1, rad2 = 1, 6 # could be also 2, 6

bad_weed_mask = bad_weed != 0
best_weed_mask = best_weed != 0

bad_weed_clean = binary_closing(bad_weed_mask, disk(rad1))
bad_weed_clean = binary_opening(bad_weed_clean, disk(rad2))

best_weed_clean = binary_closing(best_weed_mask, disk(rad1))
best_weed_clean = binary_opening(best_weed_clean, disk(rad2))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(bad_weed_clean, cmap="gray")
plt.title("Bad cleaned weed mask")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(best_weed_clean, cmap="gray")
plt.title("Best cleaned weed mask")
plt.axis('off')

plt.tight_layout()


# %%
# Testing the opencv
np.unique(bad_weed)

# %%
from skimage import measure, morphology


# %%
def show_big_obj_only(weed_mask, min_size=1000):
    label_image = measure.label(weed_mask, connectivity=2)

    filtered = morphology.remove_small_objects(label_image, min_size=min_size, connectivity=2)

    # Convert back to binary mask
    filtered_mask = (filtered > 0).astype(np.uint8) * 255

    # Show result
    plt.figure(figsize=(6,6))
    plt.imshow(filtered_mask, cmap='gray')
    plt.title("Filtered Mask (Large Blobs Only)")
    plt.axis('off')
    plt.show()

    return filtered

show_big_obj_only(bad_weed_mask);

# %%
show_big_obj_only(best_weed_mask);

# %%
bad_filtered = show_big_obj_only(bad_weed_clean, min_size=5000);

# %%
best_filtered = show_big_obj_only(best_weed_clean, min_size=5000);

# %%
def plot_labeled_image(image_data, subplot_position, title_prefix, cmap='nipy_spectral'):
    """
    Labels an image, displays it in a subplot, and sets a title.

    Args:
        image_data (np.ndarray): The input image array to be labeled.
        subplot_position (tuple): A 3-integer tuple specifying the subplot location
                                  (e.g., (1, 2, 1) for 1st subplot in a 1x2 grid).
        title_prefix (str): A prefix string for the plot title (e.g., "Best", "Bad").
        cmap (str, optional): The colormap to use for imshow. Defaults to 'nipy_spectral'.
    """
    labeled_array, num_features = ndimage.label(image_data)

    plt.subplot(*subplot_position) # Unpack the tuple for subplot arguments
    plt.imshow(labeled_array, cmap=cmap)
    plt.title(f"{title_prefix} labels with {num_features} features")
    plt.axis('off')

# %%
best_labeled_array, best_num_features = ndimage.label(best_weed_clean)
bad_labeled_array, bad_num_features = ndimage.label(bad_weed_clean)

# %%
plt.figure(figsize=(10, 5))
plot_labeled_image(bad_labeled_array, (1, 2, 1), "Bad")
plot_labeled_image(best_labeled_array, (1, 2, 2), "Best")



# %% [markdown]
# This labelled arrays of ndimage treats the majority as the background?
# 
# Edit2: Suddenly it appears to be working well.
# 

# %%
def calculate_percentage(weed_mask, land_type):
    size_image = np.prod(weed_mask.shape)
    truthy_pixel_count = np.sum(weed_mask)
    falsy_pixel_count = size_image - truthy_pixel_count
    print(f"[{land_type.upper()}] Weed section percentage: {(truthy_pixel_count / size_image):.2f}, Others section percentage: {(falsy_pixel_count / size_image):.2f}")

print("For cleaned masks:")
calculate_percentage(best_weed_clean, "best")
calculate_percentage(bad_weed_clean, "bad")

print("For uncleaned raw KMeans result masks")
calculate_percentage(best_weed_mask, "best")
calculate_percentage(bad_weed_mask, "bad")

# %% [markdown]
# As you can see, there is no way of telling which is which by only looking at the percentage values.

# %%
# Display  morphology.remove_small_objects output in plot_labeled_image
bad_filtered_labeled_array, _ = ndimage.label(bad_filtered)
best_filtered_labeled_array, _ = ndimage.label(best_filtered)
plt.figure(figsize=(10, 5))
plot_labeled_image(bad_filtered, (1, 2, 1), "Bad")
plot_labeled_image(best_filtered, (1, 2, 2), "Best")

# %%


# %%
# Now need to loop over each segments
from scipy.ndimage import find_objects, center_of_mass
slices = find_objects(best_filtered_labeled_array)
percentages = []
best_props = {}
for i, slc in enumerate(slices, start=1):
    region_mask = (best_filtered_labeled_array[slc] == i)

    area = np.sum(region_mask)
    # centroid = center_of_mass(region_mask)  # In local slice coordinates
    # centroid_global = (centroid[0] + slc[0].start, centroid[1] + slc[1].start)
   
     
    original_region = best_weed_mask[slc]
    # Sparsity
    blob = original_region * region_mask
    blob_area = np.sum(region_mask)
    blob_nonzero = np.sum(blob)
    blob_percentage = (blob_nonzero / blob_area) * 100
    percentages.append(blob_percentage)
    # best_props[i]['area'] = area
    # best_props[i]['sparsity'] = blob_percentage
    print(f"Label {i}")
    print(f"   Area of region: {area}")
    print(f"   Sparcity (in original mask): {blob_percentage:.2f} % weed inside the region")
    # print(f"   Blob area: {barea}")


# %%
# from skimage.measure import regionprops
# bad_props = regionprops(bad_labeled_array)
# for region in bad_props[:6]:
#     label_id = region.label
#     area = region.area
#     centroid = region.centroid
#     # Bounding box (min_row, min_col, max_row, max_col)
#     minr, minc, maxr, maxc = region.bbox

#     # Crop corresponding region from original image
#     original_region = bad_weed_mask[minr:maxr, minc:maxc]

#     # Region mask from morphologically cleaned (tight binary mask)
#     region_mask = region.image

#     # Sparsity: how many pixels in original_region are nonzero where region_mask is True
#     masked_original = original_region * region_mask
#     total_pixels = region_mask.size
#     nonzero_pixels = np.count_nonzero(masked_original)
#     sparsity = (1 - nonzero_pixels / total_pixels) * 100

#     print(f"Label {label_id}")
#     print(f"  Area: {area}")
#     print(f"  Centroid: {centroid}")
#     print(f"  Sparsity (in original): {sparsity:.2f}%")

# %%
slices[1]

# %%
# Extract the msavis from dict
best_msavi = raw_inputs['best']['msavi']
best_rgb = raw_inputs['best']['rgb']
bad_msavi = raw_inputs['bad']['msavi']

# %%
number = 11
loc = find_objects(best_filtered_labeled_array)[number]
print(loc)
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(best_filtered_labeled_array[loc])
plt.subplot(132)
region_mask = best_filtered_labeled_array[loc] == number + 1
print(f"Area of region: {np.sum(region_mask)}")
best_blob = best_weed_mask[loc] * region_mask
plt.imshow(best_blob, cmap="gray")
# plt.imshow(region_mask, cmap="gray")
# plt.imshow(best_weed_mask[loc], cmap="gray") # original
# let's plot the msavi masked section
plt.subplot(133)
plt.imshow(best_msavi[loc] * region_mask, cmap="viridis", vmin=0, vmax=1)
best_blob_area = np.sum(region_mask)
best_blob_nonzero = np.count_nonzero(best_blob)
print(f"Percentage of non zero section inside the blob = {(best_blob_nonzero / best_blob_area * 100):.2f} %")

# %%
def analyze_region(label_index, labeled_array, mask_array, msavi=None, plot=False):
    """
    Analyze a specific labeled region in the image.

    Parameters:
    - label_index: int, the label index to analyze (starting from 1)
    - labeled_array: 2D array, the labeled image (e.g., from labeling regions)
    - mask_array: 2D array, the binary or probability mask (e.g., vegetation mask)
    - plot: bool, whether to visualize the analysis

    Returns:
    - area: int, the number of pixels in the region
    - blob_percentage: float, percentage of mask area inside the region
    """
    assert mask_array.dtype == bool, f"Mask array is not of boolean type: dtype = {mask_array.dtype}"
    slices = find_objects(labeled_array)
    
    if label_index < 1 or label_index > len(slices):
        raise ValueError("Invalid label index provided.")

    slc = slices[label_index - 1]
    region_mask = (labeled_array[slc] == label_index)
    assert region_mask.dtype == bool, f"Region mask is not of boolean type: dtype = {region_mask.dtype}"
    area = np.sum(region_mask)

    # Apply mask
    original_region = mask_array[slc]
    blob = original_region * region_mask
    blob_nonzero = np.sum(blob)
    if msavi is not None:
        msavi_slice = msavi[slc]
        msavi_region = msavi_slice * region_mask


    if area == 0:
        blob_percentage = 0.0
    else:
        blob_percentage = (blob_nonzero / area) * 100

    if plot:
        print(f"Slice for label {label_index}: {slc}")
        print(f"Area of region: {area}")
        print(f"Percentage of non-zero pixels inside the region: {blob_percentage:.2f} %")
        
        plt.figure(figsize=(12, 5))
        plt.subplot(131)
        plt.imshow(labeled_array[slc], cmap="nipy_spectral")
        plt.title("Labeled region")
        

        plt.subplot(132)
        plt.imshow(blob, cmap="gray")
        plt.title("Masked blob")

        if msavi is not None:
            plt.subplot(133)
            plt.imshow(msavi_region, cmap="viridis")
            plt.title("MSAVI for this region")
        else:
            plt.subplot(133)
            plt.imshow(region_mask, cmap="gray")
            plt.title("Region mask")

        plt.tight_layout()
        plt.show()

    return area, blob_percentage


# %%
analyze_region(7, best_filtered_labeled_array, best_weed_mask, best_msavi, plot=True)

# %% [markdown]
# # What about for the `BAD`

# %%
for i in range(1, 6):  # for first 5 labels
    area, percent = analyze_region(i, bad_filtered_labeled_array, bad_weed_mask, plot=False)
    print(f"Label {i}:")
    print(f"   Area = {area}")
    print(f"   Weed coverage = {percent:.2f}%")

# %%
analyze_region(8, bad_filtered_labeled_array, bad_weed_mask, plot=True)

# %%
analyze_region(8, bad_filtered_labeled_array, bad_weed_mask, bad_msavi, plot=True)

# %%
analyze_region(9, bad_filtered_labeled_array, bad_weed_mask, bad_msavi, plot=True)

# %% [markdown]
# # Returning to `Best`
# Now let's take a look at `Best`
# 

# %%
# Checking whether or not the function that I typed above results in same output as the zadgai one.
for i in range(1, 6):  # for first 5 labels
    area, percent = analyze_region(i, best_filtered_labeled_array, best_weed_mask, plot=False)
    print(f"Label {i}:")
    print(f"   Area = {area}")
    print(f"   Weed coverage = {percent:.2f}%")

# %% [markdown]
# # Testing these image processing steps on different types of land images

# %%
# Make it a pipeline
# Clean the image
good_weed_mask = results['good']['masks']['weed']
plt.imshow(good_weed_mask, cmap="gray")
plt.title("Raw Kmeans segmentation mask")
plt.axis('off')
good_weed_clean = clean_mask(good_weed_mask, rad1, rad2)
good_filtered = show_big_obj_only(good_weed_clean, min_size=5000);

# %% [markdown]
# `good` looks pretty good!

# %%
worst_weed = results['worst']['masks']['weed']
worst_weed_mask = worst_weed != 0
plt.imshow(worst_weed_mask, cmap="gray")
plt.title("Raw Kmeans segmentation mask")
plt.axis('off')
worst_weed_clean = clean_mask(worst_weed_mask, rad1, rad2)
worst_filtered = show_big_obj_only(worst_weed_clean, min_size=5000);

# %%
worst_labeled_array, worst_num_features = ndimage.label(worst_filtered)
print(f"Worst num of features {worst_num_features}")
worst_msavi = raw_inputs['worst']['msavi']
analyze_region(12, worst_labeled_array, worst_weed_mask, worst_msavi, plot=True)

# %% [markdown]
# <!-- TODO: Ene yagaad hudlaa garaad bna? -->
# Fixed: always set != 0 for masks

# %%
worst_ground_mask = results['worst']['masks']['ground'] != 0
print(worst_ground_mask)
plt.imshow(worst_ground_mask, cmap="gray")

# %%
analyze_region(10, worst_labeled_array, worst_weed_mask, worst_msavi, plot=True)

# %%



