import numpy as np
import pandas as pd
import napari
import imageio.v3 as iio
from magicgui import magicgui, widgets
from pathlib import Path
import sys
from datetime import datetime
import warnings

# --- Configuration ---
N_RANDOM = 50 # Default number of points
N_PIXELS_ZOOM = 15 # Determines zoom level (approx neighborhood size)
DEFAULT_MODE = 'rgb' # Initial view mode
# Base directory containing land type subfolders (Worst_10, Bad_10, etc.)
BASE_INPUT_DIR = Path("/media/razydave/HDD/MendeFolder/BelceeriinDoroitol/reflectance_msavi_threshold/output_ref3")
# Directory to save label files
DEFAULT_OUTPUT_DIR = Path("./napari_labels") # Save in current dir by default

# --- Helper Functions ---

def get_land_type_path(base_dir: Path) -> tuple[Path | None, str | None]:
    """Prompts user for land type and returns the corresponding path and name."""
    land_types = ["worst", "bad", "medium", "good", "best"]
    folders = {
        0: "Worst_10", 1: "Bad_10", 2: "Medium_10", 3: "Good_10", 4: "Best_10"
    }

    while True:
        try:
            choice_str = input(
                "Select land type number [0: worst, 1: bad, 2: medium, 3: good, 4: best]: "
            )
            if not choice_str:
                 print("No input provided. Exiting.")
                 return None, None
            choice = int(choice_str)
            if 0 <= choice <= 4:
                land_folder_name = folders[choice]
                land_type_name = land_types[choice]
                path = base_dir / land_folder_name
                if path.exists() and path.is_dir():
                    print(f"Selected land type: {land_type_name} ({path})")
                    return path, land_type_name
                else:
                    print(f"Error: Directory not found: {path}")
                    return None, None # Exit if dir not found
            else:
                print("Invalid choice. Please enter a number between 0 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
             print(f"An error occurred: {e}")
             return None, None

def load_required_images(land_path: Path) -> dict | None:
    """Loads Red, Green, Blue, and MSAVI images from the specified path."""
    required_files = {
        # Assuming consistent naming patterns - adjust if needed
        "red": "*red_reference.tif",
        "green": "*green*.tif",
        "blue": "*blue*.tif",
        "msavi": "*msavi*.tif",
        # "ndvi": "*ndvi*.tif" # Optional: uncomment if needed
    }
    images = {}
    print(f"Searching for images in: {land_path}")

    for name, pattern in required_files.items():
        found_files = list(land_path.glob(pattern))
        if not found_files:
            print(f"Error: Could not find required '{name}' image using pattern '{pattern}' in {land_path}")
            return None
        if len(found_files) > 1:
            warnings.warn(f"Warning: Found multiple files for '{name}' using pattern '{pattern}'. Using the first one: {found_files[0].name}")
        try:
             print(f"Loading {name} from {found_files[0].name}...")
             images[name] = iio.imread(found_files[0])
             print(f"  -> Loaded {name} with shape: {images[name].shape}, dtype: {images[name].dtype}")
        except Exception as e:
             print(f"Error loading image file {found_files[0]}: {e}")
             return None

    # Validate shapes (optional but recommended)
    ref_shape = images["red"].shape
    for name, img in images.items():
         if img.shape != ref_shape:
              warnings.warn(f"Warning: Shape mismatch! {name} shape {img.shape} != reference shape {ref_shape}")
              # Decide how to handle: error out, resize, or just warn
              # print(f"Error: Image shapes do not match. {name} shape is {img.shape}, expected {ref_shape}")
              # return None # Example: Error out if shapes don't match

    return images

def main():
    # 1. Load Data
    land_path, land_name = get_land_type_path(BASE_INPUT_DIR)
    if not land_path:
        sys.exit("Exiting: No valid land type selected or path found.")
    images = load_required_images(land_path)
    if not images:
        sys.exit("Exiting: Failed to load required images.")

    # 2. Create RGB for display (if you have red, green, blue)
    rgb_stack = np.stack([images['red'], images['green'], images['blue']], axis=-1)
    rgb_norm = (rgb_stack - np.min(rgb_stack)) / (np.max(rgb_stack) - np.min(rgb_stack))  # Normalize for display

    # 3. Start napari viewer
    viewer = napari.Viewer()
    image_layer = viewer.add_image(rgb_norm, rgb=True, name='RGB Image')

    # Add Points layer for pixel selection
    points_layer = viewer.add_points(name='Selected Points', size=10)

    # Add Shapes layer for drawing polygons
    shapes_layer = viewer.add_shapes(name='Drawn Polygons', shape_type='polygon')

    # 4. Add a button to print/export selections
    @magicgui(call_button="Print Selections")
    def print_selections():
        print("Selected Points:")
        print(points_layer.data)

        print("Drawn Polygons:")
        for i, poly in enumerate(shapes_layer.data):
            print(f"Polygon {i + 1}: {poly}")

        # Optional: Save selections to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / land_name
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(output_dir / f"points_{timestamp}.csv", points_layer.data, delimiter=",", header="y,x", comments='')
        with open(output_dir / f"polygons_{timestamp}.txt", "w") as f:
            for i, poly in enumerate(shapes_layer.data):
                f.write(f"Polygon {i + 1}:\n{poly}\n\n")

        print(f"Selections saved to {output_dir}")

    viewer.window.add_dock_widget(print_selections, area="right")

    napari.run()

if __name__ == "__main__":
    main()

