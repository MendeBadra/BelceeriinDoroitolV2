# This overwrites the plots... 
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
    'worst': 0.26738,   #0.1994333333,
    'bad': 0.24646,  #0.1701666667,
    'medium': 0.1943,  #0.1698666667,
    'good': 0.0519,  #0.233,
    'best': 0.1854   #0.3486
}

class Tee:
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

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
    inds_path, land_type, threshold, output_dir_base, k_clusters = args
    msavi_image = None
    stats_dict = None
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Starting processing for {inds_path.name} (land_type: {land_type})")
        start_time = time.time()
        msavi_image = iio.imread(inds_path)
        logger.debug(f"Loaded image {inds_path.name} with shape {msavi_image.shape}")
        # Save plots in fplan directory under output_dir_base/land_type/fplan
        output_dir_imgs = output_dir_base / land_type # / "fplan" # NO need for fplan
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

def process_one_land_type_folder(land_type_folder, threshold_dict, output_dir, k_clusters=3):
    land_type_folder = Path(land_type_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Processing all fplan folders in land_type folder: {land_type_folder}")
    key = land_type_folder.name.lower()
    if "_" in key:
        key = key.split("_")[0]

    fplan_folders = [f for f in land_type_folder.iterdir() if f.is_dir() and f.name.upper().endswith("FPLAN")]
    if not fplan_folders:
        logger.warning("No fplan folders found.")
        return

    for fplan_folder in fplan_folders:
        inds_files = list(fplan_folder.glob("DJI_*INDS.TIF"))
        logger.info(f"Found {len(inds_files)} INDS files in {fplan_folder.name}")
        if not inds_files:
            logger.warning(f"No INDS files found in {fplan_folder.name}.")
            continue
        summary_csv = output_dir / f"{fplan_folder.name}_msavi_statistics.csv"

        # --- Resume logic: read already processed inds_file names ---
        processed_inds_files = set()
        if summary_csv.exists() and summary_csv.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(summary_csv)
                if "inds_file" in df_existing.columns:
                    processed_inds_files = set(df_existing["inds_file"].astype(str))
                    logger.info(f"Resuming: {len(processed_inds_files)} files already processed in {fplan_folder.name}.")
            except Exception as e:
                logger.warning(f"Could not read existing summary CSV: {e}")

        first_row = not summary_csv.exists() or summary_csv.stat().st_size == 0
        processed_count = 0
        error_count = 0
        start_time = time.time()
        with open(summary_csv, 'a', newline='') as f_csv:
            for inds_path in tqdm(inds_files, desc=f"Processing {fplan_folder.name}"):
                if inds_path.name in processed_inds_files:
                    continue
                threshold = threshold_dict.get(key, 0.2)
                if threshold == 0.2:
                    print("!!!!!!!!WARNING: NO LAND TYPE IDENTIFIED!!!!!!!!!!!!!")
                args = (inds_path, fplan_folder.name, threshold, output_dir, k_clusters)
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
        logger.info(f"Processing of {fplan_folder.name} completed in {total_time:.2f} seconds")
        logger.info(f"Successfully processed: {processed_count} images in {fplan_folder.name}")
        logger.info(f"Failed to process: {error_count} images in {fplan_folder.name}")
        if len(inds_files) > 0:
            logger.info(f"Average time per image in {fplan_folder.name}: {total_time/len(inds_files):.2f} seconds")
        logger.info(f"Summary statistics saved to: {summary_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 2025-06-11_one_land_type_compute.py <land_type_folder> <output_folder>")
        sys.exit(1)
    land_type_folder = sys.argv[1]
    output_folder = sys.argv[2]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_folder) / "logs" / f"processing_terminal_output_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"--- Python Version Check ---")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    import concurrent.futures
    print(f"concurrent.futures location: {concurrent.futures.__file__}")
    print(f"---------------------------")
    process_one_land_type_folder(
        land_type_folder, 
        THRESHOLDS, 
        output_folder, 
        k_clusters=3
    )
    tee.close()
