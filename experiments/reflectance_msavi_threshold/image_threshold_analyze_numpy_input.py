# %%
# import libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path

SUFFIX = ""
print(f"SUFFIX: {SUFFIX}")

from utils import extract_patch

# %%
assert sys.argv[1], f"No argument provided:{sys.argv[1]}" 

random_points_dir = Path(sys.argv[1])
assert random_points_dir.exists(), f"Path: {random_points_dir} doesn't exists."
ground_points = np.load(random_points_dir / "output_labels_ground.npy")
vegetation_points = np.load(random_points_dir / "output_labels_vegetation.npy")
print(f"Got the points... from {random_points_dir}, there is {len(vegetation_points)} vegetation points and {len(ground_points)} ground points.")

# %% [markdown]
# ## Load the imagestack again! smh

# %%
worst_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Worst_10"+ SUFFIX)
bad_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Bad_10"+ SUFFIX)
medium_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Medium_10"+ SUFFIX)
good_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Good_10"+ SUFFIX)
best_10_path = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/RandomForests/output/Best_10"+ SUFFIX)

which_land: int = int(input("We are now only considering October images. Please input number for one of the lands" \
                   "Possible [0, 1, 2, 3, 4] ≈ ['worst' -> 'best']: "))


assert type(which_land) == int ,f"Not an int type {which_land} -> {type(which_land)}"
assert 0 <= which_land <= 4, f"Not a valid range {which_land}"

land_10_path = [worst_10_path, bad_10_path, medium_10_path, good_10_path, best_10_path][which_land]

file_paths = [file for file in land_10_path.glob("*.tif")]
file_names = [file_path.name[:-4] for file_path in file_paths]
images = {name: iio.imread(file_path) for name, file_path in zip(file_names, file_paths)}
image_rgb = np.stack([images[channel] for channel in ("red_reference", "green_aligned", "blue_aligned")], axis=2)
plt.imshow(image_rgb)
plt.show(block=False)

# %% [markdown]
# Since now we have the `image_rgb`, we can look up every random points that we've created.

# %%
print(f"In total, there are {len(vegetation_points)}, {len(ground_points)} points")
# %%
N_PIXELS = 10  # e.g. 51x51 neighborhood
HALF = N_PIXELS // 2

random_points = np.concat([vegetation_points, ground_points])
### NOTE:TEMPORARY #
# random_points = random_points[:, [1,0]]
labels = ['v' for v in vegetation_points] + ['g' for g in ground_points]
random_labels_points = [(label, point) for point,label in zip(random_points, labels)]
# random.shuffle(random_labels_points)

random_labels_points = random_labels_points
# %%
# Create a figure with subplots in a grid
fig, axes = plt.subplots(10, 10, figsize=(20, 10))
print(f"Length of random_labels_points: {len(random_labels_points)}")
axes = axes.flatten()

for idx, (label, point) in enumerate(random_labels_points[:100]):
    print(idx)
    # Extract coordinates
    i, j = point
    neighborhood, pad_left, pad_top = extract_patch(image_rgb, idx,  i, j, HALF)    

    if np.all(neighborhood == 0):
        axes[idx].axis('off')
        axes[idx].set_title(f"Point {idx+1}: {label} (skipped)", color='gray')
        continue
    # Plot the neighborhood
    axes[idx].imshow(neighborhood)
    
    # Highlight the center pixel
    center_i = HALF - pad_left
    center_j = HALF - pad_top
    rect = plt.Rectangle((center_j - 0.5, center_j - 0.5), 1, 1, 
                       linewidth=1.5, edgecolor='red', facecolor='none')
    axes[idx].add_patch(rect)
    
    # Add label information
    title = f"Point {idx+1}: {label}"
    color = 'green' if label == 'v' else 'brown'
    axes[idx].set_title(title, color=color)
    
    # Remove ticks for cleaner look
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

plt.tight_layout()
#%%
labels
msavi_values =[]
labels_ = []
for label, point in random_labels_points:
    msavi_value = (images["msavi"][*point] / 256) * 2 - 1
    msavi_values.append(msavi_value)
    labels_.append(label)

print(f"msavi_values length: {msavi_values}")
# %% [markdown]
# Since we have our corresponding labels and msavi_values, let's plot them like a logistic regression curve.

# %%
labels_binary = [1 if label=='v' else 0 for label in labels_]
labels_binary

# %%
X = np.array(msavi_values).reshape(-1, 1)
y = np.array(labels_binary)# .reshape(-1, 1)
log_reg = LogisticRegression()
log_reg.fit(X, y)

# %%
X_new = np.linspace(-1, 1, 200).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
weight = log_reg.coef_[0][0]
bias = log_reg.intercept_[0]
decision_boundary = -bias / weight
# decision_boundary = X_new[np.argmin(np.abs(y_proba - 0.5))][0]
print(f"Decision: {decision_boundary}")
# %%

plt.plot(X_new, y_proba, "g-", label="Logistic Model")
plt.axvline(x=decision_boundary, color='k', linestyle='--', label=f"Decision at {decision_boundary:.2f}")
plt.scatter(X, y, alpha=0.6)
plt.xlabel("MSAVI Value")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show(block=False)


# %%
# ts = np.linspace(-1.0, 1.0, 200)
plt.scatter(X, y, c=y, cmap='bwr', edgecolors='k')
plt.axvline(x=decision_boundary, color='purple', linestyle='--', label="Scikit-Learn шийдвэрлэх хил")
xs = np.linspace(-1, 1, 200)
plt.plot(xs, log_reg.predict_proba(xs.reshape(-1, 1)))
plt.legend()
plt.xlabel("Бие даасан хувьсагч")
plt.ylabel("Ангилал")
plt.title("Scikit-Learn логистик регрессийн шийдвэрлэх зааг")
plt.show()
# %%
# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import sys
from pathlib import Path
import pandas as pd
import warnings

# %%
# --- Constants ---
N_PIXELS = 10  # Neighborhood diameter (e.g., 10 -> 11x11 if odd, 10x10 if even - adjusted later)
# Ensure N_PIXELS is odd for a central pixel
if N_PIXELS % 2 == 0:
    N_PIXELS += 1
    print(f"Adjusted N_PIXELS to be odd: {N_PIXELS}")
HALF = N_PIXELS // 2
N_VISUALIZE_POINTS = 100 # Max points to visualize in the patch grid
# GRID_SIZE = 10 # Grid dimensions for visualization (GRID_SIZE x GRID_SIZE)

# Suffix for potential output files (optional)
SUFFIX = ""
print(f"SUFFIX: {SUFFIX}")
print(f"Using neighborhood size: {N_PIXELS}x{N_PIXELS}")

# --- Utility Functions ---

def extract_patch(image, i, j, half_size):
    """
    Extracts a patch of size (2*half_size + 1) x (2*half_size + 1) centered at (i, j).
    Handles boundary conditions by padding conceptually (returns smaller patch if near edge).
    Returns the patch and the padding amounts applied at the top and left.
    """
    h, w = image.shape[:2]
    target_size = 2 * half_size + 1

    # Calculate bounds, ensuring they are within image dimensions
    i_start = max(0, i - half_size)
    j_start = max(0, j - half_size)
    i_end = min(h, i + half_size + 1)
    j_end = min(w, j + half_size + 1)

    patch = image[i_start:i_end, j_start:j_end]

    # Calculate padding needed if the patch couldn't be fully extracted
    pad_top = max(0, half_size - i)
    pad_left = max(0, half_size - j)
    # These pads indicate how much the *center* pixel is shifted in the *returned* patch
    # Not needed for this script's visualization, but useful generally.

    # Check if the extracted patch is smaller than the target size due to boundaries
    # This warning logic was slightly different in the original, let's keep it simple
    if patch.shape[0] < target_size or patch.shape[1] < target_size:
         # Suppress repetitive warnings if many points are near edges
        # warnings.warn(f"Point ({i, j}): Patch smaller than {target_size}x{target_size} due to image boundary. Shape: {patch.shape}", UserWarning)
        pass # Or print a less frequent summary later

    return patch # Only return the patch, padding info not used downstream here

def load_data(csv_path: Path) -> pd.DataFrame:
    """Loads the DataFrame from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    try:
        # Assuming comments might start with // or #
        df = pd.read_csv(csv_path, delimiter=',', comment='#')
        # Alternative if only // comments:
        # with open(csv_path, 'r') as f:
        #     lines = [line for line in f if not line.strip().startswith('//')]
        # df = pd.read_csv(io.StringIO('\n'.join(lines)), delimiter=',')

        print(f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")
        # Basic validation
        required_cols = ['i', 'j', 'label', 'land_type']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame missing one or more required columns: {required_cols}")
        if not ('msavi_exact' in df.columns or 'msavi_value' in df.columns):
             raise ValueError("DataFrame must contain 'msavi_exact' or 'msavi_value' column.")
        return df
    except Exception as e:
        print(f"Error loading or parsing CSV file: {csv_path}")
        raise e

def prepare_land_type_data(df_land: pd.DataFrame):
    """
    Prepares data for a specific land type: extracts coordinates, labels, and MSAVI values.
    Returns None if data is insufficient.
    """
    if df_land.empty:
        print("No data for this land_type slice.")
        return None

    vegetation_points = df_land[df_land['label'] == 'v'][['i', 'j']].values
    ground_points = df_land[df_land['label'] == 'g'][['i', 'j']].values
    print(f"Found {len(vegetation_points)} vegetation points and {len(ground_points)} ground points.")

    if len(vegetation_points) == 0 and len(ground_points) == 0:
        print("No vegetation or ground points found for this land type.")
        return None

    coordinates = df_land[['i', 'j']].values.astype(int)
    labels = df_land['label'].tolist()

    # Use MSAVI values from DataFrame
    if 'msavi_exact' in df_land.columns:
        msavi_values = df_land['msavi_exact'].values
    elif 'msavi_value' in df_land.columns:
        msavi_values = df_land['msavi_value'].values
    else:
        # This case should have been caught by load_data, but double-check
        raise ValueError("No MSAVI column found in DataFrame slice.")

    # Convert labels to binary
    if 'binary_label' in df_land.columns:
        labels_binary = df_land['binary_label'].values
    else:
        labels_binary = np.array([1 if label == 'v' else 0 for label in labels])

    # Check for sufficient data variation for modeling
    if len(np.unique(labels_binary)) < 2:
        print("Warning: Only one class present for this land type. Logistic Regression may not be meaningful.")
        # Decide if you want to proceed or return None
        # return None # Option to skip modeling if only one class

    X = msavi_values.reshape(-1, 1)
    y = labels_binary

    return {
        "X": X,
        "y": y,
        "coordinates": coordinates,
        "labels_str": labels,
        "n_veg": len(vegetation_points),
        "n_ground": len(ground_points)
    }


# --- Modeling and Analysis Functions ---

def train_and_evaluate_logistic(X, y, land_type):
    """Trains Logistic Regression, evaluates, and returns model and boundary."""
    print("\n--- Training Logistic Regression ---")
    if X is None or y is None or len(X) == 0:
         print("Insufficient data for training.")
         return None, None
    if len(np.unique(y)) < 2:
        print("Skipping Logistic Regression: Only one class present.")
        return None, None

    log_reg = LogisticRegression(solver='liblinear') # Good for smaller datasets
    log_reg.fit(X, y)

    weight = log_reg.coef_[0][0]
    bias = log_reg.intercept_[0]

    # Avoid division by zero if weight is negligible
    decision_boundary = -bias / weight if abs(weight) > 1e-6 else None
    print(f"Model: Probability = 1 / (1 + exp(-({weight:.4f} * MSAVI + {bias:.4f})))")
    if decision_boundary is not None:
        print(f"Decision boundary (P=0.5) for {land_type}: MSAVI = {decision_boundary:.4f}")
    else:
        print(f"Decision boundary calculation skipped (weight near zero: {weight:.4f})")


    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred = log_reg.predict(X)
    print(f"Confusion Matrix (Logistic Regression Raw Output for {land_type}):")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    # Pretty plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ground (0)', 'Vegetation (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Logistic Regression CM - {land_type}')
    plt.show(block=False)
    # plt.savefig(f"logreg_cm_{land_type}{SUFFIX}.png")
    # plt.close()


    print(f"\nClassification Report (Logistic Regression Raw Output for {land_type}):")
    print(classification_report(y, y_pred, target_names=['Ground (0)', 'Vegetation (1)']))

    return log_reg, decision_boundary

def plot_logistic_curve(log_reg, X, y, decision_boundary, land_type):
    """Plots the logistic regression curve and data points."""
    if log_reg is None:
        return
    print("\n--- Plotting Logistic Regression Curve ---")
    fig, ax = plt.subplots()

    # Generate points for the curve
    min_val = X.min() - 0.1
    max_val = X.max() + 0.1
    X_new = np.linspace(min_val, max_val, 300).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new) # Gets probabilities for class 0 and 1

    ax.plot(X_new, y_proba[:, 1], "g-", label="P(Vegetation | MSAVI)") # Plot probability of class 1

    # Add data points with jitter for better visualization
    jitter = np.random.normal(0, 0.02, size=y.shape)
    ax.scatter(X, y + jitter, alpha=0.5, label="Data Points (jittered)", s=10) # Smaller points

    if decision_boundary is not None:
        ax.axvline(x=decision_boundary, color='k', linestyle='--',
                   label=f"Decision Boundary ({decision_boundary:.3f})")

    ax.set_xlabel("MSAVI Value")
    ax.set_ylabel("Probability / Label")
    ax.set_title(f"Logistic Regression for {land_type}")
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_ylim(-0.1, 1.1) # Ensure 0 and 1 are clearly visible
    ax.set_xlim(min_val, max_val)
    plt.tight_layout()
    plt.show(block=False)
    # plt.savefig(f"logreg_curve_{land_type}{SUFFIX}.png")
    # plt.close(fig)

def perform_threshold_analysis(X, y, land_type):
    """
    Finds the optimal MSAVI threshold minimizing classification errors (FP+FN).
    Evaluates performance at this threshold.
    """
    print("\n--- Performing Threshold Analysis ---")
    if X is None or y is None or len(X) == 0:
         print("Insufficient data for threshold analysis.")
         return None, None
    if len(np.unique(y)) < 2:
        print("Skipping Threshold Analysis: Only one class present.")
        return None, None

    msavi_values_flat = X.flatten()
    thresholds = np.linspace(msavi_values_flat.min() - 0.01, msavi_values_flat.max() + 0.01, 201)
    results = []

    for threshold in thresholds:
        # Predict: 1 if MSAVI > threshold, else 0
        y_pred_threshold = (msavi_values_flat > threshold).astype(int)

        # Calculate confusion matrix elements directly
        # Note: scikit-learn's confusion_matrix([TN, FP], [FN, TP]) layout can be confusing.
        # Let's calculate manually for clarity:
        tp = np.sum((y == 1) & (y_pred_threshold == 1))
        tn = np.sum((y == 0) & (y_pred_threshold == 0))
        fp = np.sum((y == 0) & (y_pred_threshold == 1)) # Predicted Veg, Actual Ground
        fn = np.sum((y == 1) & (y_pred_threshold == 0)) # Predicted Ground, Actual Veg

        # Sanity check: tp + tn + fp + fn should equal len(y)
        # assert tp + tn + fp + fn == len(y)

        results.append({
            'threshold': threshold,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'total_errors': fp + fn
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("Could not generate threshold results.")
        return None, None

    # Find threshold minimizing total errors
    # Handle potential ties by choosing the first one (usually lower threshold)
    optimal_idx = results_df['total_errors'].idxmin()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    min_errors = results_df.loc[optimal_idx, 'total_errors']

    print(f"Optimal threshold (minimizing FP+FN) for {land_type}: {optimal_threshold:.4f}")
    print(f"Minimum errors (FP+FN) at this threshold: {min_errors} out of {len(y)}")


    # --- Evaluate at Optimal Threshold ---
    print(f"\n--- Evaluation at Optimal Threshold ({optimal_threshold:.4f}) ---")
    y_pred_optimal = (msavi_values_flat > optimal_threshold).astype(int)

    print(f"Confusion Matrix (Optimal Threshold for {land_type}):")
    cm_opt = confusion_matrix(y, y_pred_optimal)
    print(cm_opt)
    # Pretty plot
    disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=['Ground (0)', 'Vegetation (1)'])
    disp_opt.plot(cmap=plt.cm.Blues)
    plt.title(f'Optimal Threshold CM - {land_type} (Thr={optimal_threshold:.3f})')
    plt.show(block=False)
    # plt.savefig(f"optimal_cm_{land_type}{SUFFIX}.png")
    # plt.close()


    print(f"\nClassification Report (Optimal Threshold for {land_type}):")
    print(classification_report(y, y_pred_optimal, target_names=['Ground (0)', 'Vegetation (1)']))

    # Explicitly print TP, TN, FP, FN from the optimal confusion matrix
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel() # Correct unpacking order for scikit-learn CM
    print(f"\nTrue Positives (Veg correctly ID'd): {tp_opt}")
    print(f"True Negatives (Ground correctly ID'd): {tn_opt}")
    print(f"False Positives (Ground called Veg): {fp_opt}") # Type I Error
    print(f"False Negatives (Veg called Ground): {fn_opt}") # Type II Error
    print(f"Total Error Rate: {(fp_opt + fn_opt) / len(y):.4f}")
    print(f"\n###########################\n OPTIMAL_THRESHOLD ({land_type}): {optimal_threshold:.4f} \n###########################")


    return optimal_threshold, results_df

def plot_threshold_analysis(results_df, optimal_threshold, land_type):
    """Plots the results of the threshold analysis."""
    if results_df is None:
        return
    print("\n--- Plotting Threshold Analysis ---")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results_df['threshold'], results_df['tp'], 'g-', label='True Positives (Veg)')
    ax.plot(results_df['threshold'], results_df['tn'], 'b-', label='True Negatives (Ground)')
    ax.plot(results_df['threshold'], results_df['fp'], 'r:', label='False Positives (Ground->Veg)')
    ax.plot(results_df['threshold'], results_df['fn'], 'm:', label='False Negatives (Veg->Ground)')
    ax.plot(results_df['threshold'], results_df['total_errors'],
             'k-', linewidth=2, label='Total Errors (FP + FN)')

    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='grey', linestyle='--',
                    label=f'Optimal Threshold = {optimal_threshold:.3f}')

    ax.grid(True, linestyle=':')
    ax.legend(fontsize=9)
    ax.set_title(f'Threshold Analysis: Finding Optimal MSAVI Threshold ({land_type})')
    ax.set_xlabel('MSAVI Threshold Value')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show(block=False)
    # plt.savefig(f"threshold_analysis_{land_type}{SUFFIX}.png")
    # plt.close(fig)

# --- Main Execution ---

def run_analysis(df: pd.DataFrame):
    """Runs the analysis pipeline for each land type in the DataFrame."""
    land_types = df['land_type'].unique()
    # Optional: Define a preferred order
    order = ['worst', 'bad', 'medium', 'good', 'best']
    land_types_sorted = [lt for lt in order if lt in land_types] + \
                        [lt for lt in land_types if lt not in order] # Add any others

    print(f"\nFound land types: {land_types_sorted}")

    results_summary = {}

    for land_type in land_types_sorted:
        print(f"\n\n{'='*15} Analyzing Land Type: {land_type} {'='*15}")
        df_land = df[df['land_type'] == land_type].copy() # Use copy to avoid SettingWithCopyWarning

        # 1. Prepare Data
        prepared_data = prepare_land_type_data(df_land)
        if prepared_data is None:
            print(f"Skipping analysis for {land_type} due to insufficient data.")
            results_summary[land_type] = {'status': 'skipped - no data'}
            continue # Skip to the next land type

        X = prepared_data['X']
        y = prepared_data['y']
        coordinates = prepared_data['coordinates']
        labels_str = prepared_data['labels_str']

        # 2. Visualize Sample Patches (Locations)
        visualize_sample_patches(coordinates, labels_str, land_type)

        # 3. Train and Evaluate Logistic Regression
        log_reg, decision_boundary = train_and_evaluate_logistic(X, y, land_type)

        # 4. Plot Logistic Curve
        if log_reg:
             plot_logistic_curve(log_reg, X, y, decision_boundary, land_type)
        else:
             print("Skipping logistic curve plot as model was not trained.")


        # 5. Perform Threshold Analysis
        optimal_threshold, results_df = perform_threshold_analysis(X, y, land_type)

        # 6. Plot Threshold Analysis Results
        if results_df is not None:
            plot_threshold_analysis(results_df, optimal_threshold, land_type)
        else:
            print("Skipping threshold analysis plot.")


        # Store key results
        results_summary[land_type] = {
            'status': 'completed',
            'n_veg': prepared_data['n_veg'],
            'n_ground': prepared_data['n_ground'],
            'logistic_boundary': decision_boundary,
            'optimal_threshold': optimal_threshold
        }
        print(f"{'='*15} Finished Land Type: {land_type} {'='*15}")
    threshold_data_list = [] 
    # --- Final Summary ---
    print("\n\n{'='*20} Analysis Summary {'='*20}")
    for land_type, summary in results_summary.items():
        print(f"\nLand Type: {land_type}")
        print(f"  Status: {summary['status']}")
        if summary['status'] == 'completed':
            print(f"  Points: {summary['n_veg']} Veg, {summary['n_ground']} Ground")
            log_bound = f"{summary['logistic_boundary']:.4f}" if summary['logistic_boundary'] is not None else "N/A"
            opt_thr = f"{summary['optimal_threshold']:.4f}" if summary['optimal_threshold'] is not None else "N/A"
            print(f"  Logistic Regression Boundary (P=0.5): {log_bound}")
            print(f"  Optimal MSAVI Threshold (Min Errors): {opt_thr}")
            row_data = {
                "land_type": land_type,
                "optimal_threshold": opt_thr # Store the raw value
            }
            
            threshold_data_list.append(row_data)
            # --- <<< End Added Section >>> ---
    threshold_df = pd.DataFrame(threshold_data_list)
    # Prompt the user the name to save the threshold_df
    data_csv_dir = Path("data_csv")
    threshold_df_name = input("Enter a name for the threshold DataFrame (without extension): ")
    threshold_df_name = f"threshold_df_{threshold_df_name}.csv"
    threshold_df_path = data_csv_dir / threshold_df_name
    threshold_df.to_csv(threshold_df_path, index=False)
    print(f"\nThreshold DataFrame saved as: {threshold_df_name} in data_csv directory.")    
    print(f"{'='*58}")

    # Keep plots open until user closes them or script ends
    print("\nAll analyses complete. Plots are displayed (if any). Close plot windows to exit.")
    plt.show(block=True) # Block at the very end


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No argument provided.")
        print("Usage: python your_script_name.py <path_to_dataframe_csv>")
        sys.exit(1)

    input_df_path_str = sys.argv[1]
    input_df_path = Path(input_df_path_str)

    try:
        # Load data
        main_df = load_data(input_df_path)

        # Run the analysis pipeline
        run_analysis(main_df)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Data Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# %%
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

y_pred = log_reg.predict(X)
print(confusion_matrix(labels_binary, y_pred))
print(classification_report(y_pred, y))

# %% [markdown]
# This threshold is not kind of optimal. We have 9 instances of vegetation being misclassified into the ground region. Let's yeeball the graph and take let's say threshold value to be `0.21`

# %%
print(decision_boundary)

# %%
print("##### EYEBALL #########")
eye_balled_threshold = -0.20
y_pred_v1 = np.array((X.flatten() > eye_balled_threshold) * 1)
y_pred_v1 

# %%
print(confusion_matrix(y_pred=y_pred_v1, y_true=y))
print(classification_report(y_pred_v1, y))

# %% [markdown]
# Maybe, according to my crude experiment, `0.172` could be the good threshold value. But the two sides of misclassification is rather poor.

# %% [markdown]
# Let's do a more systematic search for the optimal threshold by testing different values
# and comparing the performance metrics

# %%
# Generate a range of threshold values from -1 to 1
thresholds = np.linspace(-1, 1, 101)  # 101 values from -1 to 1
results = []

# Calculate metrics for each threshold
for threshold in thresholds:
    y_pred_threshold = (X.flatten() > threshold).astype(int)
    
    # Calculate confusion matrix
    tp, fp, fn, tn = confusion_matrix(y_pred=y_pred_threshold, y_true=y).ravel()
    
    # Store results
    results.append({
        'threshold': threshold,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    })

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Find the threshold that minimizes the sum of false positives and false negatives
optimal_threshold = results_df.loc[(results_df['false_positive'] + results_df['false_negative']).idxmin()]['threshold']
print(f"Optimal threshold: {optimal_threshold:.2f} with errors: {results_df['false_positive'].min() + results_df['false_negative'].min()}")

# Plot the confusion matrix components vs threshold
plt.figure(figsize=(10, 6))
plt.plot(results_df['threshold'], results_df['true_positive'], 'g-', label='True Positives')
plt.plot(results_df['threshold'], results_df['true_negative'], 'b-', label='True Negatives')
plt.plot(results_df['threshold'], results_df['false_positive'] + results_df['false_negative'], 
         'r-', linewidth=2, label='Total Errors (FP + FN)')
plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
            label=f'Optimal Threshold = {optimal_threshold:.2f}')
plt.grid(True)
plt.legend()
plt.title('Finding Optimal Threshold by Minimizing Errors')
plt.xlabel('Threshold')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# Print confusion matrix and classification report for the optimal threshold
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

# %%

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

# %%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
plot_classifier(log_reg, X, y, title="Logistic Regression", label="Logistic Boundary", color='purple')

# %%
# SVM
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X, y)
plot_classifier(svm_model, X, y, title="SVM (RBF)", label="SVM Boundary", color='green')


# # %%
# # Random Forest
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier()
# rf_model.fit(X, y)
# plot_classifier(rf_model, X, y, title="Random Forest", label="Random Forest", color='orange')
# # %%
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=30)
# knn_model.fit(X, y)
# plot_classifier(knn_model, X, y, title='KNN model', label='Knn boundary', color='green')

# #%%
# # %%
# from xgboost import XGBClassifier
# xgb_model = XGBClassifier()
# xgb_model.fit(X, y)
# plot_classifier(xgb_model, X, y, title="XGBClassifier", label='XGB boundary', color='magenta')


# %%
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X, y)
plot_classifier(nb_model, X, y, title='Naive Bayes Model', label='NB Boundary', color='cyan')

# %%

df = pd.DataFrame(random_points, columns=["i", "j"])
df["label"] = labels
df["msavi_values"] = msavi_values

df = df[['label', 'i', 'j', 'msavi_values']]
save_path = Path(input("Please input the path to save the data: "))
# save_path.mkdir(exist_ok=True)
df.to_csv(save_path)