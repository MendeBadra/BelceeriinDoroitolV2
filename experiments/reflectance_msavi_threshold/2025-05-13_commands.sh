#!/bin/bash
# # filepath: /media/razydave/HDD/MendeFolder/BelceeriinDoroitol/reflectance_msavi_threshold/2025-05-13_commands.sh.sh

# # Set source and destination directories
# SRC_DIR="/media/razydave/HUSTAI_data/Khustai_Datas/2024/October"
# DEST_DIR="/media/razydave/CenAppMath/MendeCenMathApp/drone_data_2024_october"

# # Make sure destination directory exists
# mkdir -p "$DEST_DIR"

# # Process each land type directory and copy DJI_P4_MS contents
# for dir in "$SRC_DIR"/*; do
#     if [ -d "$dir" ]; then
#         # Extract the land type from directory name
#         dir_name=$(basename "$dir")
        
#         # Parse the land type (e.g., "1_Best_2024_10_01" becomes "Best_10")
#         if [[ $dir_name =~ ([0-9]+)_([A-Za-z]+)_[0-9]{4}_([0-9]{2})_([0-9]{2}) ]]; then
#             land_type="${BASH_REMATCH[2]}"
#             month="${BASH_REMATCH[3]}"
#             new_dir_name="${land_type}_${month}"
            
#             # Check if DJI_P4_MS exists in this land type folder
#             if [ -d "$dir/DJI_P4_MS" ]; then
#                 echo "Processing $dir_name → $new_dir_name"
                
#                 # Create the destination folder
#                 mkdir -p "$DEST_DIR/$new_dir_name"
                
#                 # Copy all files from DJI_P4_MS to the new destination
#                 cp -r "$dir/DJI_P4_MS/"* "$DEST_DIR/$new_dir_name/"
                
#                 echo "Copied DJI_P4_MS contents to $DEST_DIR/$new_dir_name"
#             else
#                 echo "Warning: DJI_P4_MS not found in $dir"
#             fi
#         else
#             echo "Warning: Could not parse directory name: $dir_name"
#         fi
#     fi
# done

# echo "Copy operation complete."


# just found out in CenAppMath there was actually saved drone_data_2024_october_reflectance
# Run processing commands
echo "Starting processing pipeline..."
# python batch_raw2ref.py
python align_images_batch.py
python compute_gabor_batch.py
echo "Processing complete."