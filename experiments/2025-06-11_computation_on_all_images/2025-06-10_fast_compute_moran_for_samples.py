# This script is only meant to be used with samples directory structure
import sys
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import imageio.v3 as iio
import pandas as pd
from tqdm import tqdm
import gc
import time

from align_pipeline.kmeans import main_process_msavi_for_weeds

THRESHOLDS = {
    'worst': 0.1994333333,
    'bad': 0.1701666667,
    'medium': 0.1698666667,
    'good': 0.233,
    'best': 0.3486
}

def setup_logging(output_dir):
    """Setup logging configuration with both file and console handlers."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"processing_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def process_one_image(args):
    inds_path, land_type, threshold, output_dir_base, k_clusters = args
    msavi_image = None
    stats_dict = None
    
    # Get logger for this process
    logger = logging.getLogger(__name__)

    try:
        logger.debug(f"Starting processing for {inds_path.name} (land_type: {land_type})")
        start_time = time.time()
        
        msavi_image = iio.imread(inds_path)
        logger.debug(f"Loaded image {inds_path.name} with shape {msavi_image.shape}")
        
        output_dir_imgs = output_dir_base / land_type
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
            stats_dict["inds_file"] = inds_path.name
            
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {inds_path.name} in {processing_time:.2f}s")
        
        return stats_dict

    except Exception as e:
        logger.error(f"Failed to process {inds_path}: {e}")
        return {"error_file": str(inds_path), "error_message": str(e), "land_type": land_type, "inds_file": inds_path.name}
    finally:
        if msavi_image is not None:
            del msavi_image
        gc.collect()

# def process_all_msavi_in_folder_parallel(root_dir, threshold_dict, output_dir, k_clusters=3, num_workers=None, max_tasks_per_child=None):
#     root_dir = Path(root_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Setup logging
#     logger = setup_logging(output_dir)
    
#     logger.info(f"Starting parallel processing")
#     logger.info(f"Input directory: {root_dir}")
#     logger.info(f"Output directory: {output_dir}")
#     logger.info(f"K-clusters: {k_clusters}")
#     logger.info(f"Thresholds: {threshold_dict}")
    
#     summary_csv = output_dir / "all_msavi_statistics.csv"

#     tasks = []
#     land_type_folders = [f for f in root_dir.iterdir() if f.is_dir()]
    
#     # Add progress bar for task preparation
#     for land_type_folder in tqdm(land_type_folders, desc="Scanning directories"):
#         key = land_type_folder.name.lower()
#         if "_" in key:
#             key = key.split("_")[0]
        
#         inds_files = list(land_type_folder.glob("DJI_*INDS.TIF"))
#         logger.info(f"Found {len(inds_files)} INDS files in {land_type_folder.name}")
        
#         # Add progress bar for file collection within each directory
#         for inds_path in tqdm(inds_files, desc=f"Preparing {land_type_folder.name}", leave=False):
#             threshold = threshold_dict.get(key, 0.2)
#             tasks.append((inds_path, land_type_folder.name, threshold, output_dir, k_clusters))

#     if not tasks:
#         logger.warning("No tasks found to process.")
#         return

#     logger.info(f"Prepared {len(tasks)} tasks for processing")

#     if num_workers is None:
#         num_workers = os.cpu_count() or 1
#         # Remove the artificial limit for better utilization of 12 cores
#         num_workers = min(num_workers, 10)  # Use 10 out of 12 cores, leaving 2 for system

#     executor_options = {'max_workers': num_workers}
#     if max_tasks_per_child is not None:
#         executor_options['max_tasks_per_child'] = max_tasks_per_child

#     logger.info(f"Using {executor_options.get('max_workers')} workers")
#     if 'max_tasks_per_child' in executor_options:
#         logger.info(f"Max tasks per child: {executor_options['max_tasks_per_child']}")
#     else:
#         logger.info("Max tasks per child: unlimited (or default)")

#     first_row = not summary_csv.exists() or summary_csv.stat().st_size == 0
#     processed_count = 0
#     error_count = 0
#     start_time = time.time()

#     with open(summary_csv, 'a', newline='') as f_csv:
#         with ProcessPoolExecutor(**executor_options) as executor:
#             futures = [executor.submit(process_one_image, task) for task in tasks]
#             timeout_seconds = 300
#             logger.info(f"Submitted all {len(futures)} tasks to executor")

#         # Enhanced progress bar with more information
#         progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Processing images")
#         for future in progress_bar:
#             try:
#                 stats_dict = future.result(timeout=timeout_seconds)
#             except Exception as e:
#                 logger.error(f"Unexpected error from future: {e}")
#                 stats_dict = {
#                     "error_file": "Unknown (exception)",
#                     "error_message": str(e),
#                     "land_type": "Unknown",
#                     "inds_file": "Unknown"
#                 }

#             if stats_dict is not None:
#                 if "error_file" in stats_dict:
#                     error_count += 1
#                 else:
#                     processed_count += 1
                
#                 # Update progress bar description with current stats
#                 progress_bar.set_postfix({
#                     'Success': processed_count,
#                     'Errors': error_count,
#                     'Rate': f"{processed_count + error_count}/{len(tasks)}"
#                 })
                    
#                 df_row = pd.DataFrame([stats_dict])
#                 if first_row:
#                     df_row.to_csv(f_csv, header=True, index=False)
#                     first_row = False
#                 else:
#                     df_row.to_csv(f_csv, header=False, index=False)
#                 f_csv.flush()

#     total_time = time.time() - start_time
#     logger.info(f"Processing completed in {total_time:.2f} seconds")
#     logger.info(f"Successfully processed: {processed_count} images")
#     logger.info(f"Failed to process: {error_count} images")
#     logger.info(f"Average time per image: {total_time/len(tasks):.2f} seconds")
#     logger.info(f"Summary statistics saved to: {summary_csv}")

# GEMINI go BRRR..
def process_all_msavi_in_folder_parallel(root_dir, threshold_dict, output_dir, k_clusters=3, num_workers=None, max_tasks_per_child=None):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info(f"Starting parallel processing")
    logger.info(f"Input directory: {root_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"K-clusters: {k_clusters}")
    logger.info(f"Thresholds: {threshold_dict}")
    
    summary_csv = output_dir / "all_msavi_statistics.csv"

    tasks = []
    land_type_folders = [f for f in root_dir.iterdir() if f.is_dir()]
    
    for land_type_folder in tqdm(land_type_folders, desc="Scanning directories"):
        key = land_type_folder.name.lower()
        if "_" in key:
            key = key.split("_")[0]
        
        inds_files = list(land_type_folder.glob("DJI_*INDS.TIF"))
        logger.info(f"Found {len(inds_files)} INDS files in {land_type_folder.name}")
        
        for inds_path in tqdm(inds_files, desc=f"Preparing {land_type_folder.name}", leave=False):
            threshold = threshold_dict.get(key, 0.2)
            tasks.append((inds_path, land_type_folder.name, threshold, output_dir, k_clusters))

    if not tasks:
        logger.warning("No tasks found to process.")
        return

    logger.info(f"Prepared {len(tasks)} tasks for processing")

    if num_workers is None:
        num_workers = os.cpu_count() or 1
        num_workers = min(num_workers, 10)

    executor_options = {'max_workers': num_workers}
    if max_tasks_per_child is not None:
        executor_options['max_tasks_per_child'] = max_tasks_per_child

    logger.info(f"Using {executor_options.get('max_workers')} workers")
    if 'max_tasks_per_child' in executor_options:
        logger.info(f"Max tasks per child: {executor_options['max_tasks_per_child']}")
    else:
        logger.info("Max tasks per child: unlimited (or default)")

    first_row = not summary_csv.exists() or summary_csv.stat().st_size == 0
    processed_count = 0
    error_count = 0
    start_time = time.time()

    # Corrected Logic: The loop is now INSIDE the 'with' blocks
    with open(summary_csv, 'a', newline='') as f_csv:
        with ProcessPoolExecutor(**executor_options) as executor:
            futures = [executor.submit(process_one_image, task) for task in tasks]
            logger.info(f"Submitted all {len(futures)} tasks to executor")

            progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Processing images")
            for future in progress_bar:
                try:
                    # Set a timeout for getting the result
                    stats_dict = future.result(timeout=300) 
                except Exception as e:
                    logger.error(f"A task generated an exception: {e}")
                    stats_dict = {
                        "error_file": "Unknown (exception in future)",
                        "error_message": str(e),
                        "land_type": "Unknown",
                        "inds_file": "Unknown"
                    }

                if stats_dict is not None:
                    if "error_file" in stats_dict:
                        error_count += 1
                    else:
                        processed_count += 1
                    
                    progress_bar.set_postfix({
                        'Success': processed_count,
                        'Errors': error_count
                    })
                        
                    df_row = pd.DataFrame([stats_dict])
                    if first_row:
                        df_row.to_csv(f_csv, header=True, index=False)
                        first_row = False
                    else:
                        df_row.to_csv(f_csv, header=False, index=False)
                    # Flush the buffer to ensure the row is written to disk immediately
                    f_csv.flush()

    total_time = time.time() - start_time
    logger.info(f"Processing completed in {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {processed_count} images")
    logger.info(f"Failed to process: {error_count} images")
    if len(tasks) > 0:
        logger.info(f"Average time per image: {total_time/len(tasks):.2f} seconds")
    logger.info(f"Summary statistics saved to: {summary_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 2025-06-10_fast_compute_moran_for_samples.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    print(f"--- Python Version Check ---")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    import concurrent.futures
    print(f"concurrent.futures location: {concurrent.futures.__file__}")
    print(f"---------------------------")

    process_all_msavi_in_folder_parallel(
        input_folder, 
        THRESHOLDS, 
        output_folder, 
        k_clusters=3,
        num_workers=6,  # Utilize more cores for your 12-core CPU
        max_tasks_per_child=5  # Reduce to better manage memory with more workers
    )
