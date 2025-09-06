import os

import pandas as pd
import numpy as np

# TODO: try to understand it and refactor it.
def delete_zero_vi_rows(dir_path: str, delete_targets, end_str="", ext=".npy") -> None:
    print(f"Targets found: {delete_targets.shape[0]}")
    for index, target in delete_targets.iterrows():
        filename = os.path.join(dir_path, f"patch_{target['start_vert']}_{target['end_vert']}_{target['start_horiz']}_{target['end_horiz']}{'' if end_str == '' else '_'+end_str}{ext}")
        if os.path.exists(filename):
            os.remove(filename)
            print(f"{index}: patch_{target['start_vert']}_{target['end_vert']}_{target['start_horiz']}_{target['end_horiz']} has been deleted.")
        else:
            print(f"{index}: patch_{target['start_vert']}_{target['end_vert']}_{target['start_horiz']}_{target['end_horiz']} doesn't exist\n filename: {filename}")
            
            
def clean_dir_from_blanks(gabor_some_dir: str, delete_targets) -> None:    
    localhist_dir = os.path.join(gabor_some_dir, 'LocalHistEq_Mask')
    delete_zero_vi_rows(localhist_dir, delete_targets, end_str='local_eq_gabor_mask')
    msavi_dir = os.path.join(gabor_some_dir, 'MSAVI_Mask')
    delete_zero_vi_rows(msavi_dir, delete_targets, end_str="MSAVI_mask")
    plot_dir = os.path.join(gabor_some_dir, 'Original_and_Local_Hist_Equalized')
    delete_zero_vi_rows(plot_dir, delete_targets, ext=".png")
    orig_gray_dir = os.path.join(gabor_some_dir, 'OriginalGrayscale_Mask')
    delete_zero_vi_rows(orig_gray_dir, delete_targets, end_str="original_gabor_mask")
    
    print("Finished job.")
    
def process_data(df: pd.DataFrame, field: str):
    """
        A function to remove NaNs in the MSAVI column and put NDVI values in the same row. It removes the unnecessary columns and adds new columns.
        The dataframe argument must be from the output of the gabor_segmentation process.
    """
    df['class'] = field
    df_msavi = df[df['VegetationIndex'] == 'MSAVI'].reset_index(drop=True)
    columns_nan = df_msavi.columns[df_msavi.isna().any()].tolist()
    df_ndvi = df[df['VegetationIndex'] == 'NDVI'].reset_index(drop=True)

    df_msavi = df_msavi.drop(columns=columns_nan)
    df_msavi = df_msavi.join(df_ndvi[columns_nan], rsuffix='_ndvi')
    df_msavi['patch_number'] = df_msavi.apply(
        lambda row: f"{row['start_vert']}_{row['end_vert']}_{row['start_horiz']}_{row['end_horiz']}",
        axis=1
    )
    df_msavi['AreaSum'] = df_msavi['VI_Area'] + df_msavi['Pure_GaborArea']
    df_msavi['Unknown_Area'] = df_msavi['Patch_Area'] - df_msavi['AreaSum']
    df_msavi = df_msavi.drop(['start_vert', 'end_vert', 'start_horiz', 'end_horiz', 'patch'], axis=1)
    df_msavi['Normalized_VI_Area'] = df_msavi['VI_Area'] / df_msavi['Patch_Area']
    df_msavi['Normalized_Pure_GaborArea'] = df_msavi['Pure_GaborArea'] / df_msavi['Patch_Area']
    df_msavi['Normalized_Unknown_Area'] = df_msavi['Unknown_Area'] / df_msavi['Patch_Area']

    df_msavi = df_msavi[['patch_number', 'VegetationIndex', 'Patch_Area', 'VI_Area',
                         'Pure_GaborArea', 'AreaSum', 'Unknown_Area', 'class',
                         'Normalized_VI_Area', 'Normalized_Pure_GaborArea',
                         'Normalized_Unknown_Area']]
    return df_msavi

