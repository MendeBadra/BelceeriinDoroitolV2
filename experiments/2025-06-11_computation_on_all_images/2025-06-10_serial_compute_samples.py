# This overwrites the plots... (FIXED IT)


import sys
import os
import logging
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
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"processing_{timestamp}.log"
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
    """Expected args: (inds_path, land_type, threshold, output_dir_base, k_clusters)"""
    inds_path, land_type, threshold, output_dir_base, k_clusters = args
    msavi_image = None
    stats_dict = None
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Starting processing for {inds_path.name} (land_type: {land_type})")
        start_time = time.time()
        msavi_image = iio.imread(inds_path)
        logger.debug(f"Loaded image {inds_path.name} with shape {msavi_image.shape}")
        output_dir_imgs = output_dir_base / land_type #  May be this?
        output_dir_imgs.mkdir(parents=True, exist_ok=True)
        file_prefix = inds_path.stem[:8]
        ######## MAIN FUNCTION CALL ########
        print(f"Processing {inds_path.name} with threshold {threshold} and k_clusters {k_clusters}")
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

def process_all_msavi_in_folder_serial(root_dir, threshold_dict, output_dir, k_clusters=3):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Starting serial processing")
    logger.info(f"Input directory: {root_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"K-clusters: {k_clusters}")
    logger.info(f"Thresholds: {threshold_dict}")
    summary_csv = output_dir / "all_msavi_statistics.csv" # DON'T EXIST??????
    tasks = []
    # --- Resume logic: read already processed inds_file names ---
    # ----------

    land_type_folders = [f for f in root_dir.iterdir() if f.is_dir()]
    for land_type_folder in tqdm(land_type_folders, desc="Scanning directories"):
        key = land_type_folder.name.lower()
        if "_" in key:
            key = key.split("_")[0]
        inds_files = list(land_type_folder.glob("DJI_*INDS.TIF"))
        logger.info(f"Found {len(inds_files)} INDS files in {land_type_folder.name}")
        for inds_path in tqdm(inds_files, desc=f"Preparing {land_type_folder.name}", leave=False):
            threshold = threshold_dict.get(key, 0.2)
            if threshold == 0.2:
                print("!!!!!WARNING: NO LAND_TYPE IDENTIFIED!!!!!!")
            tasks.append((inds_path, land_type_folder.name, threshold, output_dir, k_clusters))
    if not tasks:
        logger.warning("No tasks found to process.")
        return
    logger.info(f"Prepared {len(tasks)} tasks for processing")
    first_row = not summary_csv.exists() or summary_csv.stat().st_size == 0
    processed_count = 0
    error_count = 0
    start_time = time.time()
    with open(summary_csv, 'a', newline='') as f_csv:
        for args in tqdm(tasks, desc="Processing images"):
            stats_dict = process_one_image(args)
            if stats_dict is not None:
                if "error_file" in stats_dict:
                    error_count += 1
                else:
                    processed_count += 1
                df_row = pd.DataFrame([stats_dict])
                if first_row:
                    df_row.to_csv(f_csv, header=True, index=False)
                    first_row = False
                else:
                    df_row.to_csv(f_csv, header=False, index=False)
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
    process_all_msavi_in_folder_serial(
        input_folder, 
        THRESHOLDS, 
        output_folder, 
        k_clusters=3
    )