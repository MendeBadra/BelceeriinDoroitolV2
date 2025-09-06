from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

# Assume these are defined elsewhere, e.g., in a config file or earlier in your script
# ALIGNED_ROOT_DIR = Path("path/to/your/aligned/data")
# THRESHOLDS = {"landtype1": 0.5, "default": 0.3} # Example
# PLOT_SAVE_DIR = Path("path/to/save/plots/and/csv")
# Path to the main root directory for aligned imagery
ALIGNED_ROOT_DIR = Path("/path/to/your/ALIGNED_ROOT_DIR") # Placeholder - replace with actual path
THRESHOLDS = {"default": 0.25} # Placeholder - replace/extend with your actual thresholds
PLOT_SAVE_DIR = Path("/path/to/your/PLOT_SAVE_DIR") # Placeholder - replace with actual path
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure plot save directory exists


# >>> This is the function you would need to define or modify <<<
# >>> See section 2 below for a detailed example of this function <<<
# def process_all_task(task_args):
#     # drone_image_number, land_type, threshold, fplan_dir, save_plots_flag = task_args
#     # ... main processing logic for INDS file ...
#     # ... ADD RGB HANDLING HERE using fplan_dir, drone_image_number, and save_plots_flag ...
#     # (example provided in the next section)
#     # return results_dict
#     pass







if __name__ == "__main__":
    tasks = []
    for category_dir in Path(ALIGNED_ROOT_DIR).iterdir():
        if category_dir.is_dir():
            land_type_from_folder = category_dir.name.lower().split('_')[0]
            threshold = THRESHOLDS.get(land_type_from_folder, THRESHOLDS.get("default"))
            if threshold is None:
                print(f"Skipping {category_dir.name}: No threshold found for land_type '{land_type_from_folder}' and no default.")
                continue

            for fplan_dir in category_dir.iterdir():
                if fplan_dir.is_dir():
                    # Assuming INDS.TIF contains or is related to the RGB reflectance data
                    for inds_path in fplan_dir.glob("DJI_*INDS.TIF"):
                        drone_image_number = inds_path.stem.replace("DJI_", "").replace("INDS", "")
                        # Add task: (drone_image_number, land_type, threshold, fplan_dir, save_plots_flag, inds_path)
                        # We now pass inds_path directly to process_all_task
                        # The save_plots_flag (set to True) can signal process_all_task to save the RGB visualization
                        tasks.append((drone_image_number, land_type_from_folder, threshold, fplan_dir, True, inds_path))

    print(f"Generated {len(tasks)} tasks.")

    if not tasks:
        print("No tasks to process. Check ALIGNED_ROOT_DIR, THRESHOLDS, and file naming conventions.")
    else:
        num_workers = 4 # IMPORTANT: Adjust based on your system's memory and CPU. Start small.
        print(f"Running tasks with up to {num_workers} workers...")

        all_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # process_all_task will be defined to handle the tuple including inds_path
            all_results = list(tqdm(executor.map(process_all_task, tasks), total=len(tasks)))

        results_df = pd.DataFrame([res for res in all_results if isinstance(res, dict)])

        if not results_df.empty:
            print("\nSample of results:")
            print(results_df.head())

            output_csv_path = PLOT_SAVE_DIR / "processing_summary_results.csv"
            results_df.to_csv(output_csv_path, index=False)
            print(f"\nProcessing complete. Results saved to {output_csv_path}")

            # Basic error summary
            if 'critical_error' in results_df.columns:
                errors_df = results_df[results_df['critical_error'].notna()]
                if not errors_df.empty:
                    print(f"\nEncountered {len(errors_df)} critical errors during processing.")
                    print(errors_df[['drone_image_number', 'fplan', 'critical_error']].head())
            # Add summary for RGB processing errors if you add such a column
            if 'rgb_processing_error' in results_df.columns:
                rgb_errors_df = results_df[results_df['rgb_processing_error'].notna()]
                if not rgb_errors_df.empty:
                    print(f"\nEncountered {len(rgb_errors_df)} RGB processing errors.")
                    # print(rgb_errors_df[['drone_image_number', 'fplan', 'rgb_processing_error']].head())


            # ... (rest of your error reporting) ...
        else:
            print("No results were processed or all tasks failed before returning a dictionary.")