import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from pathlib import Path


# input_root = "/media/razydave/CenAppMath/MendeCenMathApp/drone_data_2024_october_reflectance"  # Adjusted path to your input directory
input_root = "/media/razydave/CenAppMath/2025_07_Hustai_Drone/" # Odoo baihgui c umnu n baisan
print(Path(input_root).exists())
# backup copy хийхээс залхуурав. Ердийн үед битгий ингээрэй

# quit()
output_root = "/media/razydave/CenAppMath/best_raw/"
script_path = "p4m/raw2ref.py"  # Adjusted path to your script

# Gather all plan folders
plan_folders = []

for category in os.listdir(input_root):
    category_path = os.path.join(input_root, category)
    if not os.path.isdir(category_path):
        continue

    for plan_folder in os.listdir(category_path):
        input_path = os.path.join(category_path, plan_folder)
        rel_path = os.path.relpath(input_path, input_root)
        output_path = os.path.join(output_root, rel_path)
        plan_folders.append((input_path, output_path))

# Worker function to run raw2ref
def process_folder(args):
    input_path, output_path = args
    os.makedirs(output_path, exist_ok=True)
    result = subprocess.run(["python", script_path, input_path, output_path])
    return input_path, result.returncode

# Run in parallel
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_folder, args) for args in plan_folders]
    for f in tqdm(as_completed(futures), total=len(futures), desc="Processing image folders"):
        input_path, returncode = f.result()
        if returncode != 0:
            print(f"[ERROR] Processing failed for {input_path}")

