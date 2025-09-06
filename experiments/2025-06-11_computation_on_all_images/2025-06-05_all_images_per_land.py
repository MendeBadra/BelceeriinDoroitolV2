

# import os
# from pathlib import Path

# import imageio.v3 as iio
# import pandas as pd
# import matplotlib.pyplot as plt

# import importlib
# from pathlib import Path
# import shutil

# import align_pipeline.kmeans
# importlib.reload(align_pipeline.kmeans)
# from align_pipeline.kmeans import main_process_msavi_for_weeds

# THRESHOLDS = {
#     'worst': 0.1994333333,
#     'bad': 0.1701666667,
#     'medium': 0.1698666667,
#     'good': 0.233,
#     'best': 0.3486
# }


# aligned_folder = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/reflectance_msavi_threshold/drone_data_2024_october_aligned")
# # output_folder = Path("random-2025-05-27_kmeans_new_fixed_calib")
# # dump to test
# # output_folder = Path("test")
# output_folder = Path("all_images_per_land")
# output_folder.mkdir(exist_ok=True)


# def process_all_msavi_in_folder(root_dir, threshold_dict, output_dir, k_clusters=3):
#     """
#     Recursively process all DJI_???INDS.TIF files in the folder structure using main_process_msavi_for_weeds.
#     Loads the INDS.TIF using imageio.v3 and supplies the array to the main function.
#     Collects all statistics in a DataFrame and saves it to the top of output_folder.
#     """
#     root_dir = Path(root_dir)
#     output_dir = Path(output_dir)
#     all_stats = []

#     for land_type_folder in root_dir.iterdir():
#         key = land_type_folder.name.lower()
#         print(f"Key = {key}")
#         if "_" in key:
#             key = key.split("_")[0]
#         if not land_type_folder.is_dir():
#             continue
#         for fplan_folder in land_type_folder.iterdir():
#             if not fplan_folder.is_dir():
#                 continue
#             inds_files = list(fplan_folder.glob("DJI_*INDS.TIF"))
#             for inds_path in inds_files:
                
#                 threshold = threshold_dict.get(key, 0.2)
#                 file_prefix = inds_path.stem[:8]
#                 print(f"Processing {inds_path} with threshold {threshold}")
#                 try:
#                     msavi_image = iio.imread(inds_path)
#                 except Exception as e:
#                     print(f"Failed to load {inds_path}: {e}")
#                     continue
#                 output_dir_imgs = output_dir / land_type_folder.name / fplan_folder.name
#                 output_dir_imgs.mkdir(parents=True, exist_ok=True)
#                 stats_dict = main_process_msavi_for_weeds(
#                     msavi_source=msavi_image,
#                     threshold=threshold,
#                     #output_dir=output_dir / land_type_folder.name / fplan_folder.name,
#                     output_dir=output_dir_imgs,
#                     file_prefix=file_prefix,
#                     k_clusters=k_clusters
#                 )
#                 if stats_dict is not None:
#                     stats_dict["land_type"] = land_type_folder.name
#                     stats_dict["fplan_folder"] = fplan_folder.name
#                     stats_dict["inds_file"] = inds_path.name
#                     all_stats.append(stats_dict)

#     # Save all statistics to a single CSV at the top of output_dir
#     if all_stats:
#         df = pd.DataFrame(all_stats)
#         summary_csv = output_dir / "all_msavi_statistics.csv"
#         df.to_csv(summary_csv, index=False)
#         print(f"All summary statistics saved to: {summary_csv}")
#     else:
#         print("No statistics collected.")
#     del all_stats, df, stats_dict, msavi_image, inds_files, output_dir_imgs

# process_all_msavi_in_folder(aligned_folder, THRESHOLDS, output_folder, k_clusters=3)

# FAULTY VERSION
# from concurrent.futures import ProcessPoolExecutor
# from pathlib import Path
# import imageio.v3 as iio
# import pandas as pd
# from tqdm import tqdm
# import gc
# import align_pipeline.kmeans
# from align_pipeline.kmeans import main_process_msavi_for_weeds

# THRESHOLDS = {
#     'worst': 0.1994333333,
#     'bad': 0.1701666667,
#     'medium': 0.1698666667,
#     'good': 0.233,
#     'best': 0.3486
# }

# aligned_folder = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/reflectance_msavi_threshold/drone_data_2024_october_aligned")
# output_folder = Path("all_images_per_land")
# output_folder.mkdir(exist_ok=True)

# def process_one_image(args):
#     inds_path, land_type, fplan_folder, threshold, output_dir, k_clusters = args
#     try:
#         msavi_image = iio.imread(inds_path)
#     except Exception as e:
#         print(f"Failed to load {inds_path}: {e}")
#         return None
#     output_dir_imgs = output_dir / land_type / fplan_folder.name
#     output_dir_imgs.mkdir(parents=True, exist_ok=True)
#     file_prefix = inds_path.stem[:8]
#     stats_dict = main_process_msavi_for_weeds(
#         msavi_source=msavi_image,
#         threshold=threshold,
#         output_dir=output_dir_imgs,
#         file_prefix=file_prefix,
#         k_clusters=k_clusters
#     )
#     if stats_dict is not None:
#         stats_dict["land_type"] = land_type
#         stats_dict["fplan_folder"] = fplan_folder.name
#         stats_dict["inds_file"] = inds_path.name
#     return stats_dict

# def process_all_msavi_in_folder_parallel(root_dir, threshold_dict, output_dir, k_clusters=3):
#     root_dir = Path(root_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     summary_csv = output_dir / "all_msavi_statistics.csv"
#     tasks = []
#     for land_type_folder in root_dir.iterdir():
#         key = land_type_folder.name.lower()
#         if "_" in key:
#             key = key.split("_")[0]
#         if not land_type_folder.is_dir():
#             continue
#         for fplan_folder in land_type_folder.iterdir():
#             if not fplan_folder.is_dir():
#                 continue
#             inds_files = list(fplan_folder.glob("DJI_*INDS.TIF"))
#             for inds_path in inds_files:
#                 threshold = threshold_dict.get(key, 0.2)
#                 tasks.append((inds_path, land_type_folder.name, fplan_folder, threshold, output_dir, k_clusters))
#     print(f"Processing {len(tasks)} images in parallel...")

#     first_row = True
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         for stats_dict in tqdm(executor.map(process_one_image, tasks), total=len(tasks), desc="Processing images"):
#             if stats_dict is not None:
#                 df = pd.DataFrame([stats_dict])
#                 df.to_csv(summary_csv, mode='a', header=first_row, index=False)
#                 first_row = False
#             del stats_dict
#             gc.collect()

#     print(f"All summary statistics saved to: {summary_csv}")
    
# # Run the parallel processing
# process_all_msavi_in_folder_parallel(aligned_folder, THRESHOLDS, output_folder, k_clusters=3)

# GEMINI go BRRRRRR.....
import sys
import os # For cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import imageio.v3 as iio
import pandas as pd
from tqdm import tqdm
import gc

# Assuming align_pipeline.kmeans and main_process_msavi_for_weeds are correctly importable
from align_pipeline.kmeans import main_process_msavi_for_weeds

THRESHOLDS = {
    'worst': 0.1994333333,
    'bad': 0.1701666667,
    'medium': 0.1698666667,
    'good': 0.233,
    'best': 0.3486
}

aligned_folder = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/reflectance_msavi_threshold/drone_data_2024_october_aligned")

output_folder = Path("all_images_per_land_mem_safe_v3")
output_folder.mkdir(exist_ok=True, parents=True)

def process_one_image(args):
    inds_path, land_type, fplan_folder, threshold, output_dir_base, k_clusters = args
    msavi_image = None
    stats_dict = None

    try:
        msavi_image = iio.imread(inds_path)
        output_dir_imgs = output_dir_base / land_type / fplan_folder.name
        output_dir_imgs.mkdir(parents=True, exist_ok=True)
        file_prefix = inds_path.stem[:8]

        stats_dict = main_process_msavi_for_weeds(
            msavi_source=msavi_image,
            threshold=threshold,
            output_dir=output_dir_imgs,
            file_prefix=file_prefix,
            k_clusters=k_clusters
        )

        if stats_dict is not None:
            stats_dict["land_type"] = land_type
            stats_dict["fplan_folder"] = fplan_folder.name
            stats_dict["inds_file"] = inds_path.name
        
        return stats_dict

    except Exception as e:
        print(f"Worker {os.getpid()} failed to process {inds_path}: {e}") # Added worker PID
        return {"error_file": str(inds_path), "error_message": str(e), "land_type": land_type, "fplan_folder": fplan_folder.name, "inds_file": inds_path.name}
    finally:
        if msavi_image is not None:
            del msavi_image
        gc.collect()


# Corrected function signature and usage of max_tasks_per_child
def process_all_msavi_in_folder_parallel(root_dir, threshold_dict, output_dir, k_clusters=3, num_workers=None, max_tasks_per_child=None):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "all_msavi_statistics.csv"

    tasks = []
    for land_type_folder in root_dir.iterdir():
        if not land_type_folder.is_dir():
            continue
        
        key = land_type_folder.name.lower()
        if "_" in key:
            key = key.split("_")[0]
        
        for fplan_folder in land_type_folder.iterdir():
            if not fplan_folder.is_dir():
                continue
            
            inds_files = list(fplan_folder.glob("DJI_*INDS.TIF"))
            for inds_path in inds_files:
                threshold = threshold_dict.get(key, 0.2)
                tasks.append((inds_path, land_type_folder.name, fplan_folder, threshold, output_dir, k_clusters))

    if not tasks:
        print("No tasks found to process.")
        return

    print(f"Preparing to process {len(tasks)} images in parallel...")

    if num_workers is None:
        num_workers = os.cpu_count() or 1
        num_workers = min(num_workers, 4) 

    executor_options = {'max_workers': num_workers}
    if max_tasks_per_child is not None: # Check if the argument was passed
        executor_options['max_tasks_per_child'] = max_tasks_per_child # Correct argument name

    # Print a confirmation of the options being used
    print(f"Using {executor_options.get('max_workers')} workers. ", end="")
    if 'max_tasks_per_child' in executor_options:
        print(f"Max tasks per child: {executor_options['max_tasks_per_child']}")
    else:
        print("Max tasks per child: unlimited (or default)")


    first_row = not summary_csv.exists() or summary_csv.stat().st_size == 0

    with open(summary_csv, 'a', newline='') as f_csv:
        with ProcessPoolExecutor(**executor_options) as executor: # Use dictionary unpacking
            futures = [executor.submit(process_one_image, task) for task in tasks]
            
            timeout_seconds = 300  # 5 minutes per task

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                stats_dict = future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                print("A task exceeded the time limit and was skipped.")
                stats_dict = {
                    "error_file": "Unknown (timeout)",
                    "error_message": f"Timeout after {timeout_seconds} seconds",
                    "land_type": "Unknown",
                    "fplan_folder": "Unknown",
                    "inds_file": "Unknown"
                }
            except Exception as e:
                print(f"Unexpected error: {e}")
                stats_dict = {
                    "error_file": "Unknown (exception)",
                    "error_message": str(e),
                    "land_type": "Unknown",
                    "fplan_folder": "Unknown",
                    "inds_file": "Unknown"
                }

            if stats_dict is not None:
                df_row = pd.DataFrame([stats_dict])
                if first_row:
                    df_row.to_csv(f_csv, header=True, index=False)
                    first_row = False
                else:
                    df_row.to_csv(f_csv, header=False, index=False)
                f_csv.flush()

    print(f"All summary statistics saved to: {summary_csv}")


if __name__ == "__main__":
    print(f"--- Python Version Check ---")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    import concurrent.futures
    print(f"concurrent.futures location: {concurrent.futures.__file__}")
    print(f"---------------------------")

    process_all_msavi_in_folder_parallel(
        aligned_folder, 
        THRESHOLDS, 
        output_folder, 
        k_clusters=3,
        num_workers=4,
        max_tasks_per_child=10  # Correct argument name
    )