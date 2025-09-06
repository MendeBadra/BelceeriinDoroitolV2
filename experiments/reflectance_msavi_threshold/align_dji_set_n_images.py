"""
This file is originally from a WeedsGalore trying out in the path:
/media/razydave/HDD/MendeFolder/git_clones/weedsgalore/khustai_align.py

I had changed the name of this script to better reflect added functionality of arbitrary length input folder images into correct output.
2025.07.28
"""


import subprocess
from pathlib import Path
import glob
import os
import re

def run_align_on_dji_sets(input_dir, output_dir):
    """
    For each set of DJI_XXX?.TIF images in input_dir, call align_dji_images.py with the set as input.
    Each set is expected to have DJI_XXX1.TIF, DJI_XXX2.TIF, ..., DJI_XXX5.TIF, where XXX is a number from 001 to 999.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # Find all TIFs matching DJI_XXX?.TIF where XXX is 001-999 and last digit is 1-5
    all_tifs = sorted(input_dir.glob("DJI_*.TIF"))
    # Group by prefix (e.g., DJI_001)
    from collections import defaultdict
    sets = defaultdict(list)
    pattern = re.compile(r"^DJI_(\d{3})([1-5])\.TIF$", re.IGNORECASE)
    for tif in all_tifs:
        m = pattern.match(tif.name)
        if m:
            prefix = f"DJI_{m.group(1)}"
            sets[prefix].append(tif)
    # Only keep sets with all 5 bands (1-5)
    complete_sets = {}
    for k, v in sets.items():
        # Check that all 5 bands (1-5) are present
        bands = set()
        for tif in v:
            m = pattern.match(tif.name)
            if m:
                bands.add(m.group(2))
        if bands == set(str(i) for i in range(1,6)):
            # Sort files by band number for consistency
            v_sorted = sorted(v, key=lambda t: int(pattern.match(t.name).group(2)))
            complete_sets[k] = v_sorted
    if not complete_sets:
        print("No complete sets of 5 bands found in", input_dir)
        return
    for prefix, files in complete_sets.items():
        # Create a temp directory for this set
        temp_input_dir = input_dir / f"{prefix}_temp"
        temp_input_dir.mkdir(exist_ok=True)
        # Symlink or copy the 5 files into temp_input_dir
        for tif in files:
            dest = temp_input_dir / tif.name
            if not dest.exists():
                try:
                    os.symlink(tif.resolve(), dest)
                except Exception:
                    # fallback to copy if symlink fails
                    import shutil
                    shutil.copy2(tif, dest)
        # Output dir for this set
        set_output_dir = output_dir / f"{prefix}_aligned"
        set_output_dir.mkdir(parents=True, exist_ok=True)
        # Call the align_dji_images.py script
        cmd = [
            "python",
            "/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/experiments/reflectance_msavi_threshold/align_dji_images.py",
            str(temp_input_dir),
            str(set_output_dir)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        # Clean up temp_input_dir
        for tif in temp_input_dir.glob("*.TIF"):
            tif.unlink()
        temp_input_dir.rmdir()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python khustai_align.py <input_dir> <output_dir>")
        exit(1)
    run_align_on_dji_sets(sys.argv[1], sys.argv[2])
