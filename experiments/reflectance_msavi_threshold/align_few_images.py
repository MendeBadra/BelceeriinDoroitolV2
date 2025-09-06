import sys
import os
from pathlib import Path
import shutil
import subprocess
from glob import glob
from tempfile import TemporaryDirectory

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_align.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find all images matching DJI_*.TIF and sort by number
    images = sorted(glob(os.path.join(input_dir, "DJI_*.TIF")),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    if len(images) % 5 != 0:
        print("Number of images is not a multiple of 5.")
        sys.exit(1)

    for i in range(0, len(images), 5):
        with TemporaryDirectory() as temp_dir:
            batch = images[i:i+5]
            for img in batch:
                shutil.copy(img, temp_dir)
            # Call align_dji_images.py with temp_dir and output_dir
            subprocess.run([
                sys.executable, "align_dji_images.py", temp_dir, str(output_dir / str(i))
            ], check=True)

if __name__ == "__main__":
    main()