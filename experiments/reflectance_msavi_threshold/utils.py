"""utility functions the has been useful in align_images_opencv.ipynb notebook"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import imageio.v3 as iio
from sklearn.metrics import confusion_matrix, classification_report


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
    (cc, warp_matrix) = cv2.findTransformECC(reference_image, target_image, warp_matrix, warp_mode, criteria)
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        aligned = cv2.warpPerspective(target_image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        aligned = cv2.warpAffine(target_image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return aligned, warp_matrix

def align_with_orb(ref_img1: np.ndarray, img2:np.ndarray): # -> warp_img, homography_matrix
    """Align images using ORB feature matching and homography"""
    # Create ORB detector
    MAX_NUM_FEATURES = 500
    # Create ORB detector
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    # Detect keypoints on the 8-bit images
    keypoints1, descriptors1 = orb.detectAndCompute(ref_img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * 0.1)
    matches = matches[:num_good_matches]

    # Step Find homography
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    height, width = img2.shape
    warped_img = cv2.warpPerspective(img2, homography, (width, height))
    return warped_img, homography


####### FROM image_threshold_analyze_runs.ipynb
def extract_patch(image, idx, i, j, half_size):
    h, w = image.shape[:2]
    i_start = max(0, i - half_size)
    j_start = max(0, j - half_size)
    i_end = min(h, i + half_size + 1)
    j_end = min(w, j + half_size + 1)
    
    patch = image[i_start:i_end, j_start:j_end]
    
    pad_left = max(0, half_size - i)
    pad_right = max(0, i + half_size + 1 - w)
    pad_top = max(0, half_size - j)
    pad_bottom = max(0, j + half_size + 1 - h)
    
    if patch.shape[:2] != (2 * half_size + 1, 2 * half_size + 1):
        print(f"Warning at Point {idx+1} ({i, j}): Unexpected patch shape {patch.shape}")
    else:
        #print(f"regular_patch shape idx: {idx}= {patch.shape}")
        pass
    return patch, pad_left, pad_top

def plot_classifier(model, X, y, title="Classifier decision", label="Model boundary", color='black'):
    # Create a range of values to visualize decision function
    X_new = np.linspace(-1, 1, 200).reshape(-1, 1)
    
    # Get predicted probabilities (if available)
    try:
        y_proba = model.predict_proba(X_new)[:, 1]
        has_proba = True
    except:
        y_proba = model.predict(X_new)
        has_proba = False

    # Plot probabilities or predictions
    plt.figure(figsize=(6, 4))
    if has_proba:
        plt.plot(X_new, y_proba, color=color, label=label)
    else:
        plt.plot(X_new, y_proba, color=color, linestyle='--', label=label)

    # Plot data points
    plt.scatter(X[y==0], y[y==0], color="blue", edgecolor='k', label="Ground", marker='o')
    plt.scatter(X[y==1], y[y==1], color="red", edgecolor='k', label="Vegetation", marker='o')
    # Find closest index to 0.5
    boundary_index = np.argmin(np.abs(y_proba - 0.5))
    decision_boundary = X_new[boundary_index][0]
    y_pred = model.predict(X)
    print(f"Confusion matrix: \n{confusion_matrix(y, y_pred)}")
    # Add to plot
    plt.axvline(decision_boundary, color='gray', linestyle='--', label='Decision Boundary')
    # Annotate
    plt.xlabel("MSAVI or Feature")
    plt.ylabel("Probability / Prediction")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.show()

def load_and_display_image(random_points):
    # Define paths for the different land types
    worst_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Worst_10")
    bad_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Bad_10")
    medium_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Medium_10")
    good_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Good_10")
    best_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Best_10")
    
    # Ask user which land to load
    which_land: int = int(input("We are now only considering October images. Please input number for one of the lands" \
                       "Possible [0, 1, 2, 3, 4] ≈ ['worst' -> 'best']: "))

    # Validate input
    assert type(which_land) == int, f"Not an int type {which_land} -> {type(which_land)}"
    assert 0 <= which_land <= 4, f"Not a valid range {which_land}"

    # Select the appropriate path based on user input
    land_10_path = [worst_10_path, bad_10_path, medium_10_path, good_10_path, best_10_path][which_land]

    # Load all TIFF files in the selected path
    file_paths = [file for file in land_10_path.glob("*.tif")]
    file_names = [file_path.name[:-4] for file_path in file_paths]
    images = {name: iio.imread(file_path) for name, file_path in zip(file_names, file_paths)}
    
    # Create RGB image from individual bands
    image_rgb = np.stack([images[channel] for channel in ("red_reference", "green_aligned", "blue_aligned")], axis=2)
    
    # Display the image with points overlaid
    plt.imshow(image_rgb)
    plt.scatter(random_points[:, 1], random_points[:, 0], color='red', marker='s', s=1)
    plt.title("Image RGB")
    plt.axis('off')
    plt.show()
    
    return images, image_rgb, random_points

def plot_labeled_points(random_labels_points, image_rgb, half):
    # Create a figure with enough subplots to display all points
    n_points = len(random_labels_points)
    n_cols = 10
    n_rows = (n_points + n_cols - 1) // n_cols  # Ceiling division to ensure enough rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2*n_rows))
    axes = axes.flatten()

    for idx, (label, point) in enumerate(random_labels_points):
        if idx >= len(axes):
            print(f"Warning: Not enough subplots for all points. Displaying only the first {len(axes)} points.")
            break
            
        # Extract coordinates
        i, j = point
        neighborhood, pad_left, pad_top = extract_patch(image_rgb, idx, i, j, half)    

        if np.all(neighborhood == 0):
            axes[idx].axis('off')
            axes[idx].set_title(f"Point {idx+1}: {label} (skipped)", color='gray')
            continue
            
        # Plot the neighborhood
        axes[idx].imshow(neighborhood)
        
        # Highlight the center pixel
        center_i = half - pad_left
        center_j = half - pad_top
        rect = plt.Rectangle((center_i - 0.5, center_j - 0.5), 1, 1, 
                           linewidth=1.5, edgecolor='red', facecolor='none')
        axes[idx].add_patch(rect)
        
        # Add label information
        title = f"Point {idx+1}: {label}"
        color = 'green' if label == 'v' else 'brown'
        axes[idx].set_title(title, color=color)
        
        # Remove ticks for cleaner look
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].axis('off')

    # Hide any unused subplots
    for idx in range(len(random_labels_points), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig, axes

def find_optimal_threshold(X, y,
                           threshold_range=(-1, 1),
                           num_thresholds=101,
                           title='Finding Optimal Threshold by Minimizing Errors'):
    """
    Find the optimal threshold for a binary classification problem by minimizing errors.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature values (must be 1D or 2D with shape (n_samples, 1))
    y : numpy.ndarray
        True binary labels (0 or 1)
    threshold_range : tuple, default=(-1, 1)
        Range of thresholds to test (min, max)
    num_thresholds : int, default=101
        Number of threshold values to test
    
    Returns:
    --------
    dict
        Contains 'optimal_threshold', 'results_df', and 'threshold_fig'
    """
    # import matplotlib.pyplot as plt
    
    # Ensure X is flattened if it's 2D
    X_flat = X.flatten() if X.ndim > 1 else X
    
    # Generate threshold range
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    results = []

    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred_threshold = (X_flat > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true=y, y_pred=y_pred_threshold)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases where all predictions are one class
            if len(np.unique(y_pred_threshold)) == 1:
                if y_pred_threshold[0] == 1:  # All predicted as positive
                    tp = np.sum(y == 1)
                    fp = np.sum(y == 0)
                    fn, tn = 0, 0
                else:  # All predicted as negative
                    tn = np.sum(y == 0)
                    fn = np.sum(y == 1)
                    tp, fp = 0, 0
        
        # Store results
        results.append({
            'threshold': threshold,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'total_error': fp + fn,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        })

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Find the threshold that minimizes the sum of false positives and false negatives
    optimal_threshold = results_df.loc[results_df['total_error'].idxmin()]['threshold']
    y_pred_optimal = (X.flatten() > optimal_threshold).astype(int)

    print(f"\nConfusion Matrix at optimal threshold ({optimal_threshold:.3f}):")
    cm = confusion_matrix(y_true=y, y_pred=y_pred_optimal)
    print(cm)

    print(f"\nClassification Report at optimal threshold ({optimal_threshold:.3f}):")
    print(classification_report(y_true=y, y_pred=y_pred_optimal))

    # Get the specific TP, TN, FP, FN values
    tp, fp, fn, tn = cm.ravel()
    print(f"\nTrue Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Total Error Rate: {(fp + fn) / len(y):.4f}")


    # Plot the confusion matrix components vs threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['threshold'], results_df['true_positive'], 'g-', label='True Positives')
    ax.plot(results_df['threshold'], results_df['true_negative'], 'b-', label='True Negatives')
    ax.plot(results_df['threshold'], results_df['total_error'], 
             'r-', linewidth=2, label='Total Errors (FP + FN)')
    ax.axvline(x=optimal_threshold, color='k', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    
    return {
        'optimal_threshold': optimal_threshold,
        'results_df': results_df,
        'threshold_fig': fig
    }

def plot_msavi_(threshold_df, land_images):
    # Create a dictionary to store thresholded images
    height, width = land_images['best']['msavi'].shape
    thresholded_images = {}
    # Create a figure with 5 rows (one for each land type) and 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    for i, land_type in enumerate(land_images.keys()):
        images = land_images[land_type]
        
        # Get the threshold value for this land type from threshold_df
        threshold_value = threshold_df.loc[threshold_df['land_type'] == land_type, 'optimal_threshold'].values[0]
        
        rgb_image = np.stack([images['red'], images['green'], images['blue']], axis=2)
        msavi_image = images['msavi']
        
        # Convert MSAVI from uint8 to float and normalize to 0-1 range
        msavi_float = (msavi_image.astype(float) / 255.0) * 2 - 1
        
        # Apply threshold
        binary_mask = (msavi_float > threshold_value).astype(np.uint8) * 255
        
        # Store the thresholded image
        thresholded_images[land_type] = binary_mask
        
        axes[i, 0].imshow(msavi_image, cmap='RdYlGn')
        axes[i, 0].set_title(f'{land_type.capitalize()} - Original MSAVI')
        axes[i, 0].set_title(f'{land_type.capitalize()} - RGB Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(binary_mask, cmap='binary')
        axes[i, 1].set_title(f'Veg. coverage: {(np.sum(binary_mask / 255)/ (height*width) * 100):.1f}%\nThreshold > {threshold_value:.2f}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    return thresholded_images

def plot_rgb_(threshold_df, land_images):
        height, width = land_images['best']['msavi'].shape
        thresholded_images = {}
        
        # Create a figure with 5 rows (one for each land type) and 2 columns
        fig, axes = plt.subplots(5, 2, figsize=(12, 20))
        
        # Loop through each land type and create the visualizations
        for i, land_type in enumerate(land_images.keys()):
            images = land_images[land_type]
            
            # Get the threshold value for this land type from threshold_df

            print(i, land_type, threshold_df.loc[threshold_df['land_type'] == land_type, 'optimal_threshold'])
            threshold_value = threshold_df.loc[threshold_df['land_type'] == land_type, 'optimal_threshold'].values[0]
            
            rgb_image = np.stack([images['red'], images['green'], images['blue']], axis=2)
            msavi_image = images['msavi']
            
            # Convert MSAVI from uint8 to float and normalize to 0-1 range
            msavi_float = (msavi_image.astype(float) / 255.0) * 2 - 1
            
            # Apply threshold
            binary_mask = (msavi_float > threshold_value).astype(np.uint8) * 255
            
            # Store the thresholded image
            thresholded_images[land_type] = binary_mask
            
            # Display the original RGB and thresholded image side by side
            axes[i, 0].imshow(rgb_image)
            axes[i, 0].set_title(f'{land_type.capitalize()} - RGB Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(binary_mask, cmap='binary')
            axes[i, 1].set_title(f'Veg. coverage: {(np.sum(binary_mask / 255)/ (height*width) * 100):.1f}%\nThreshold > {threshold_value:.2f}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        return thresholded_images

def plot(threshold_df, land_images, by='msavi'):
    """ Just to escape the notebook scope"""
    # Create a dictionary to store thresholded images
    if by == 'msavi':
        plot_msavi_(threshold_df, land_images)
    elif by == 'rgb':
        plot_rgb_(threshold_df, land_images)
    # For each land type in land_images

def plot_msavi_masked(threshold_df, land_images, by='msavi'):
    height, width = land_images['best']['msavi'].shape
    thresholded_images = {}
    fig, axes = plt.subplots(5, 2, figsize=(12, 30))
    for i, land_type in enumerate(land_images.keys()):
        images = land_images[land_type]
        threshold_value = threshold_df.loc[threshold_df['land_type'] == land_type, 'optimal_threshold'].values[0]
        
        # Get the MSAVI image for this land type
        msavi_image = images['msavi']
        
        # Convert MSAVI from uint8 to float and normalize to 0-1 range
        msavi_float = (msavi_image.astype(float) / 255.0) * 2 - 1
        
        # Apply threshold
        binary_mask = (msavi_float > threshold_value).astype(np.uint8) * 255
        # Store the thresholded image
        thresholded_images[land_type] = binary_mask
        msavi_masked = msavi_image * (binary_mask / 255.0)

        cmap = cm.get_cmap('RdYlGn')
        cmap.set_bad(color='black')
    
        if by == "msavi":
            axes[i, 0].imshow(msavi_image, cmap='RdYlGn')
        elif by == "rgb":
            rgb_image = np.stack([images['red'], images['green'], images['blue']], axis=2)
            axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title(f'{land_type.capitalize()} - Original MSAVI')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(msavi_masked, cmap=cmap)
        axes[i, 1].set_title(f'{land_type.capitalize()} - Veg. coverage: {(np.sum(binary_mask / 255)/ (height*width) * 100):.1f}%\nThresholded (>{threshold_value})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_dataframe_from_dict(land_types):
    # Initialize a list to store all points data
    all_points_data = []
    # Loop through all directories and load data
    for land_type, dirs in land_types.items():
        for dir_path in dirs:
            dir_name = dir_path.name
            
            try:
                # Load vegetation and ground points
                veg_file = dir_path / "output_labels_vegetation.npy"
                ground_file = dir_path / "output_labels_ground.npy"
                
                if veg_file.exists() and ground_file.exists():
                    veg_points = np.load(veg_file)
                    ground_points = np.load(ground_file)
                    
                    # Add vegetation points to the dataframe with their label and land type
                    for point in veg_points:
                        all_points_data.append({
                            'i': point[0],
                            'j': point[1],
                            'label': 'v',
                            'land_type': land_type,
                            'directory': dir_name
                        })
                    
                    # Add ground points to the dataframe with their label and land type
                    for point in ground_points:
                        all_points_data.append({
                            'i': point[0],
                            'j': point[1],
                            'label': 'g',
                            'land_type': land_type,
                            'directory': dir_name
                        })
                    
                    print(f"Loaded from {dir_name}: {len(veg_points)} vegetation points, {len(ground_points)} ground points")
                else:
                    print(f"Missing data files in {dir_name}")
            except Exception as e:
                print(f"Error loading data from {dir_name}: {e}")
    # Create a pandas DataFrame with all points
    df = pd.DataFrame(all_points_data)
    return df

def enrich_with_pixel_values(df: pd.DataFrame, land_images: dict) -> pd.DataFrame:
    """
    Enrich a DataFrame with MSAVI and band values based on land type and (i, j) coordinates.
    
    Parameters:
    - df: DataFrame with columns ['i', 'j', 'land_type']
    - land_images: dict where keys are land_type and values are dicts with keys:
        'msavi', 'red', 'green', 'blue', 'red_edge', 'nir' (each a 2D array)
    
    Returns:
    - A new DataFrame with added columns:
        ['msavi_value', 'msavi_exact', 'red', 'green', 'blue', 'red_edge', 'nir']
    """
    df = df.copy()

    # Initialize columns
    msavi_exact_values = []
    msavi_int_values = []
    band_values = {band: [] for band in ['red', 'green', 'blue', 'red_edge', 'nir']}

    for _, row in df.iterrows():
        i, j = row['i'], row['j']
        land_type = row['land_type']

        # MSAVI
        msavi_exact = land_images[land_type]['msavi']
        msavi_img = land_images[land_type]['msavi_scaled']
        if 0 <= i < msavi_img.shape[0] and 0 <= j < msavi_img.shape[1]:
            msavi_val = msavi_exact[i, j]
            msavi_int = msavi_img[i, j]
        else:
            msavi_val = None
            msavi_int = None
        msavi_exact_values.append(msavi_val)
        msavi_int_values.append(msavi_int)

        # Bands
        for band in band_values:
            band_img = land_images[land_type][band]
            if 0 <= i < band_img.shape[0] and 0 <= j < band_img.shape[1]:
                band_val = band_img[i, j]
            else:
                band_val = None
            band_values[band].append(band_val)

    # Add MSAVI
    df['msavi_exact'] = msavi_exact_values
    df['msavi_value'] = msavi_int_values
    # df['msavi_value'] = df['msavi_exact'].apply(lambda v: (v / 255) * 2 - 1 if v is not None else None)

    # Add bands
    for band, vals in band_values.items():
        df[band] = vals

    return df

def load_data_from_napari(napari_dir_path):
    data_csv_dir = Path("data_csv")
    df_example = pd.read_csv(data_csv_dir / "data_all_lands_v2.csv")
    # load the files into new dataframe
    df = pd.DataFrame(None, columns=df_example.columns)

    # Find all relevant directories and group them by land type
    land_paths = {
        'worst': [d for d in napari_dir_path.glob("worst_*")],
        'bad': [d for d in napari_dir_path.glob("bad_*")],
        'medium': [d for d in napari_dir_path.glob("medium_*")],
        'good': [d for d in napari_dir_path.glob("good_*")],
        'best': [d for d in napari_dir_path.glob("best_*")]
    }

    for land_name, paths in land_paths.items():
        for file in paths:
            # read the file
            labels = pd.read_csv(file)
            labels["land_type"] = land_name
            labels["directory"] = str(file.parent.name)
            labels["binary_label"] = labels["label"] == 'v'
            df = pd.concat([df, labels], axis=0, ignore_index=True)
    return df

def calculate_vegetation_coverage(threshold_df: pd.DataFrame, land_images: dict) -> pd.DataFrame:
    """
    Calculates the vegetation coverage percentage for each land type based on 
    its MSAVI image and a given threshold.

    Assumes the input MSAVI image is uint8 scaled from 0-255, representing
    the typical MSAVI range (e.g., -1 to 1). If the input MSAVI image is already
    in the float -1 to 1 range, the normalization step should be adjusted/removed.

    Args:
        threshold_df: DataFrame with columns 'land_type' (str) and 
                      'optimal_threshold' (float).
        land_images: Dictionary where keys are land types (str, matching those
                     in threshold_df) and values are dictionaries containing at
                     least the 'msavi' image as a NumPy array (uint8). 
                     Example: {'best': {'msavi': np.array(...), ...}, ...}

    Returns:
        pd.DataFrame: A DataFrame with columns 'land_type' and 
                      'vegetation_percentage'. Returns an empty DataFrame if
                      input land_images is empty or no valid calculations 
                      could be performed.
    """
    
    results = []
    
    # Check if land_images is empty
    if not land_images:
        print("Warning: Input land_images dictionary is empty.")
        return pd.DataFrame(columns=['land_type', 'vegetation_percentage'])

    for land_type, images in land_images.items():
        # --- Input Validation ---
        # Check if land_type exists in threshold_df
        threshold_row = threshold_df[threshold_df['land_type'] == land_type]
        if threshold_row.empty:
            print(f"Warning: No threshold found for land type '{land_type}' in threshold_df. Skipping.")
            continue
            
        # Check if 'msavi' key exists and is a numpy array
        if 'msavi' not in images or not isinstance(images['msavi'], np.ndarray):
            print(f"Warning: 'msavi' image not found or invalid for land type '{land_type}'. Skipping.")
            continue

        msavi_image = images['msavi']
        
        # Check if image is empty
        if msavi_image.size == 0:
             print(f"Warning: MSAVI image for land type '{land_type}' is empty. Skipping.")
             continue

        # --- Calculation ---
        try:
            threshold_value = threshold_row['optimal_threshold'].iloc[0]
            
            height, width = msavi_image.shape[:2] # Use shape[:2] for robustness with multi-channel images if any
            total_pixels = height * width

            if total_pixels == 0:
                print(f"Warning: Image for land type '{land_type}' has zero pixels. Skipping.")
                continue

            # Convert MSAVI from uint8 (0-255 assumption) to float and normalize to -1 to 1 range
            # If your msavi_image is already float -1 to 1, remove this normalization line
            msavi_float = (msavi_image.astype(float) / 255.0) * 2.0 - 1.0
            # msavi_float = msavi_image
            
            # Apply threshold to create a boolean mask (True where vegetation)
            binary_mask = msavi_float > threshold_value
            
            # Count vegetation pixels (where mask is True)
            veg_pixels = np.sum(binary_mask)
            
            # Calculate percentage
            veg_percentage = (veg_pixels / total_pixels) * 100.0
            
            # Store result
            results.append({'land_type': land_type, 'vegetation_percentage': veg_percentage})

        except IndexError:
             # This might happen if iloc[0] fails unexpectedly, though covered by threshold_row.empty check
             print(f"Error retrieving threshold value for land type '{land_type}'. Skipping.")
             continue
        except Exception as e:
            print(f"An error occurred processing land type '{land_type}': {e}. Skipping.")
            continue

    # --- Create Final DataFrame ---
    if not results:
         print("Warning: No vegetation percentages could be calculated.")
         return pd.DataFrame(columns=['land_type', 'vegetation_percentage'])
         
    coverage_df = pd.DataFrame(results)
    return coverage_df

def stretch_band(band, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.percentile(band, 2)
    if max_val is None:
        max_val = np.percentile(band, 98)
    stretched = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    return stretched

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