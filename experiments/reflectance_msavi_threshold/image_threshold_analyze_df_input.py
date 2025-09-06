# %%
# import libraries
import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path

from utils import extract_patch

SUFFIX = ""
# %%
N_PIXELS = 10  # e.g. 51x51 neighborhood
HALF = N_PIXELS // 2

print(f"SUFFIX: {SUFFIX}")


# %%
assert len(sys.argv) > 1, "No argument provided. Please provide path to DataFrame CSV"
input_df_path = Path(sys.argv[1])
assert input_df_path.exists(), f"Path: {input_df_path} doesn't exist."


# Load the dataframe, ignoring comment lines starting with //
df = pd.read_csv('data_csv/data_imgset1_1315_rgbnre.csv', delimiter=',')
print(f"Loaded DataFrame with {len(df)} rows and columns: {df.columns.tolist()}")

# Extract vegetation and ground points
vegetation_points = df[df['label'] == 'v'][['i', 'j']].values
ground_points = df[df['label'] == 'g'][['i', 'j']].values
print(f"Found {len(vegetation_points)} vegetation points and {len(ground_points)} ground points.")

# %%
def analyze_land_types(df):
    land_types = df['land_type'].unique()
    # Sort land_types if possible (from worst to best)
    order = ['worst', 'bad', 'medium', 'good', 'best']
    land_types = [lt for lt in order if lt in land_types]
    print(f"Analyzing land types: {land_types}")

    for land_type in land_types:
        print(f"\n=== Analyzing land_type: {land_type} ===")
        df_land = df[df['land_type'] == land_type]
        if df_land.empty:
            print(f"No data for land_type: {land_type}")
            continue

        # Extract points and labels
        vegetation_points = df_land[df_land['label'] == 'v'][['i', 'j']].values
        ground_points = df_land[df_land['label'] == 'g'][['i', 'j']].values
        print(f"Found {len(vegetation_points)} vegetation points and {len(ground_points)} ground points.")

        # Use points and labels from the dataframe
        random_points = np.vstack([vegetation_points, ground_points]) if len(vegetation_points) and len(ground_points) else np.array([])
        labels = df_land['label'].tolist()
        random_labels_points = [(label, point) for label, point in zip(labels, df_land[['i', 'j']].values)]

        # Plot patches if there are points
        n_points = min(len(random_labels_points), 100)
        if n_points > 0:
            fig, axes = plt.subplots(10, 10, figsize=(20, 10))
            axes = axes.flatten()
            # Dummy image for patch extraction
            image_rgb = np.zeros((max(df_land['i'])+HALF+1, max(df_land['j'])+HALF+1, 3), dtype=np.uint8)
            # for idx, (label, point) in enumerate(random_labels_points[:n_points]):
            #     i, j = point
            #     neighborhood, pad_left, pad_top = extract_patch(image_rgb, idx, int(i), int(j), HALF)
            #     if np.all(neighborhood == 0):
            #         axes[idx].axis('off')
            #         axes[idx].set_title(f"Point {idx+1}: {label} (skipped)", color='gray')
            #         continue
            #     axes[idx].imshow(neighborhood)
            #     center_i = HALF - pad_left
            #     center_j = HALF - pad_top
            #     rect = plt.Rectangle((center_j - 0.5, center_j - 0.5), 1, 1, 
            #                        linewidth=1.5, edgecolor='red', facecolor='none')
            #     axes[idx].add_patch(rect)
            #     title = f"Point {idx+1}: {label}"
            #     color = 'green' if label == 'v' else 'brown'
            #     axes[idx].set_title(title, color=color)
            #     axes[idx].set_xticks([])
            #     axes[idx].set_yticks([])
            for idx in range(n_points, 100):
                axes[idx].axis('off')
            plt.tight_layout()
            plt.suptitle(f"Neighborhoods for {land_type}", y=1.02)
            plt.show(block=False)

        # Use MSAVI values from DataFrame
        if 'msavi_exact' in df_land.columns:
            msavi_values = df_land['msavi_exact'].tolist()
        elif 'msavi_value' in df_land.columns:
            msavi_values = df_land['msavi_value'].tolist()
        else:
            raise ValueError("No MSAVI column found in DataFrame.")

        # Convert labels to binary
        if 'binary_label' in df_land.columns:
            labels_binary = df_land['binary_label'].tolist()
        else:
            labels_binary = [1 if label=='v' else 0 for label in labels]

        X = np.array(msavi_values).reshape(-1, 1)
        y = np.array(labels_binary)
        if len(X) == 0 or len(y) == 0:
            print(f"No data for land_type: {land_type}")
            continue

        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        X_new = np.linspace(-1, 1, 200).reshape(-1, 1)
        y_proba = log_reg.predict_proba(X_new)
        weight = log_reg.coef_[0][0]
        bias = log_reg.intercept_[0]
        decision_boundary = -bias / weight
        print(f"Decision boundary for {land_type}: {decision_boundary}")

        plt.figure()
        plt.plot(X_new, y_proba, "g-", label="Logistic Model")
        plt.axvline(x=decision_boundary, color='k', linestyle='--', label=f"Decision at {decision_boundary:.2f}")
        plt.scatter(X, y, alpha=0.6)
        plt.xlabel("MSAVI Value")
        plt.ylabel("Probability")
        plt.title(f"Logistic Regression for {land_type}")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

        from sklearn.metrics import classification_report, confusion_matrix

        y_pred = log_reg.predict(X)
        print(confusion_matrix(y, y_pred))
        print(classification_report(y, y_pred))

        # Threshold analysis
        thresholds = np.linspace(-1, 1, 101)
        results = []
        for threshold in thresholds:
            y_pred_threshold = (X.flatten() > threshold).astype(int)
            tp, fp, fn, tn = confusion_matrix(y, y_pred_threshold).ravel()
            results.append({
                'threshold': threshold,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            })
        results_df = pd.DataFrame(results)
        optimal_threshold = results_df.loc[(results_df['false_positive'] + results_df['false_negative']).idxmin()]['threshold']
        print(f"Optimal threshold for {land_type}: {optimal_threshold:.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['true_positive'], 'g-', label='True Positives')
        plt.plot(results_df['threshold'], results_df['true_negative'], 'b-', label='True Negatives')
        plt.plot(results_df['threshold'], results_df['false_positive'] + results_df['false_negative'], 
                 'r-', linewidth=2, label='Total Errors (FP + FN)')
        plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
                    label=f'Optimal Threshold = {optimal_threshold:.2f}')
        plt.grid(True)
        plt.legend()
        plt.title(f'Finding Optimal Threshold by Minimizing Errors ({land_type})')
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show(block=False)

        y_pred_optimal = (X.flatten() > optimal_threshold).astype(int)
        print(f"\nConfusion Matrix at optimal threshold ({optimal_threshold:.3f}):")
        cm = confusion_matrix(y, y_pred_optimal)
        print(cm)
        print(f"\nClassification Report at optimal threshold ({optimal_threshold:.3f}):")
        print(classification_report(y, y_pred_optimal))
        tp, fp, fn, tn = cm.ravel()
        print(f"\nTrue Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Total Error Rate: {(fp + fn) / len(y):.4f}")

        print(f"###########################\n OPTIMAL_THRESHOLD: {optimal_threshold:.2f} \n###########################")

# %%
if __name__ == "__main__":
    # Iterate over each unique land_type and call analyze_land_types on each slice
    for land_type in df['land_type'].unique():
        print(f"\n--- Running analysis for land_type: {land_type} ---")
        analyze_land_types(df[df['land_type'] == land_type])
