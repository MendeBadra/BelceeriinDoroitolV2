# Temuujin code for plotting the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc


def take_random_samples(df, x1, x2, target_class, sample_size = 30):
    samples_df = df[df['class'] == target_class].sample(sample_size).reset_index(drop = True)
    samples_df_mean = samples_df[[x1, x2]].mean()
    return samples_df_mean.values

def take_n_random_samples(df, x1, x2, target_class, N = 20, sample_size = 30):
    random_samples = []
    for _ in range(N):
        random_samples.append(take_random_samples(df, x1, x2, target_class, sample_size = sample_size))
        
    random_samples = pd.DataFrame(random_samples, columns = [x1, x2])
    random_samples['class'] = target_class
    random_samples['N'] = N
    random_samples['sample_size'] = sample_size
    random_samples['label'] = f"C{target_class},{sample_size},{N}"
    return random_samples

def return_random_samples(df, x1, x2, target_classes, all_N, all_sample_sizes):
    mean_df = pd.DataFrame()
    
    base_colors = {
        'worst': (127, 0, 0),  # red
        'bad': (0, 0, 127),  # blue
        'medium': (127, 127, 0),  # Yellow
        'good': (31, 31, 31), # Gray
        'best': (0, 127, 0)  # Green
    }
    
    colors = {}
    
    for target_class in target_classes:
        base_color = base_colors.get(target_class, (0, 0, 0))
        
        for i, (sample_size, N) in enumerate(zip(all_sample_sizes, all_N)):
            intensity_factor = 63*i
            color = tuple(min(255, c + intensity_factor) for c in base_color)
            key = f'C{target_class},{sample_size},{N}'
            colors[key] = color
    
    for target_class in tqdm(target_classes):
        for N, sample_size in zip(all_N, all_sample_sizes):
            random_samples = take_n_random_samples(df, x1, x2, target_class, N = N, sample_size = sample_size)
            mean_df = pd.concat([mean_df, random_samples])
    # This is where I changed the colors dictionary because of the plot I was making.
    # colors = {
    #     # Class 10
    #     f'C10,{all_sample_sizes[0]},{all_N[0]}': (0, 127, 0), 
    #     f'C10,{all_sample_sizes[1]},{all_N[1]}': (0, 191, 0), 
    #     f'C10,{all_sample_sizes[2]},{all_N[2]}': (0, 255, 0),
    #     # Class 9
    #     f'C9,{all_sample_sizes[0]},{all_N[0]}': (0, 0, 127), 
    #     f'C9,{all_sample_sizes[1]},{all_N[1]}': (63, 63, 255), 
    #     f'C9,{all_sample_sizes[2]},{all_N[2]}': (191, 191, 255),
    #     # Class 5
    #     f'C5,{all_sample_sizes[0]},{all_N[0]}': (127, 0, 0), 
    #     f'C5,{all_sample_sizes[1]},{all_N[1]}': (255, 63, 63), 
    #     f'C5,{all_sample_sizes[2]},{all_N[2]}': (255, 191, 191),
    #     # Class 2
    #     f'C2,{all_sample_sizes[0]},{all_N[0]}': (127, 127, 0),  # Yellow with increasing saturation
    #     f'C2,{all_sample_sizes[1]},{all_N[1]}': (191, 191, 0), 
    #     f'C2,{all_sample_sizes[2]},{all_N[2]}': (255, 255, 0),
    #     # Class 4
    #     f'C4,{all_sample_sizes[0]},{all_N[0]}': (31, 31, 31),  # Black with increasing saturation (darker grays)
    #     f'C4,{all_sample_sizes[1]},{all_N[1]}': (95, 95, 95), 
    #     f'C4,{all_sample_sizes[2]},{all_N[2]}': (159, 159, 159),
    # }
    
    # Mendee added colors dict
    # colors = {
    #     # Best ~~ 10
    #     f'C{target_classes[-1]},{all_sample_sizes[0]},{all_N[0]}': (0, 127, 0), 
    #     f'C{target_classes[-1]},{all_sample_sizes[1]},{all_N[1]}': (0, 191, 0), 
    #     f'C{target_classes[-1]},{all_sample_sizes[2]},{all_N[2]}': (0, 255, 0),
    #     #Bad ~~ 9
    #     f'C{target_classes[1]},{all_sample_sizes[0]},{all_N[0]}': (0, 0, 127), # blue
    #     f'C{target_classes[1]},{all_sample_sizes[1]},{all_N[1]}': (63, 63, 255), 
    #     f'C{target_classes[1]},{all_sample_sizes[2]},{all_N[2]}': (191, 191, 255),
        
    #     # Medium ~~ 2
    #     f'C{target_classes[2]},{all_sample_sizes[0]},{all_N[0]}': (127, 127, 0),  # Yellow with increasing saturation
    #     f'C{target_classes[2]},{all_sample_sizes[1]},{all_N[1]}': (191, 191, 0), 
    #     f'C{target_classes[2]},{all_sample_sizes[2]},{all_N[2]}': (255, 255, 0),
        
    #     #Good~~ 4
    #     f'C{target_classes[-2]},{all_sample_sizes[0]},{all_N[0]}': (31, 31, 31),  # Black with increasing saturation (darker grays)
    #     f'C{target_classes[-2]},{all_sample_sizes[1]},{all_N[1]}': (95, 95, 95), 
    #     f'C{target_classes[-2]},{all_sample_sizes[2]},{all_N[2]}': (159, 159, 159),
    #     #If medium arrives, add.
    #     # Worst ~~ 5
    #     f'C{target_classes[0]},{all_sample_sizes[0]},{all_N[0]}': (127, 0, 0),  # red
    #     f'C{target_classes[0]},{all_sample_sizes[1]},{all_N[1]}': (255, 63, 63), 
    #     f'C{target_classes[0]},{all_sample_sizes[2]},{all_N[2]}': (255, 191, 191),
        
    # }
    print(colors)
    colors_normalized = {key: tuple(val / 255 for val in rgb) for key, rgb in colors.items()}

    return mean_df, colors_normalized

def plot_mean_df_scatter(df, x1, x2, colors, hue, title) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, num = 1, clear = True, figsize = (16, 6))
    # Suppose the dataframe is generated from the mean of random samples from 
    # the original dataframe. This function is typically used with the return_random_samples function.
    for ax in (ax1, ax2):
        sns.scatterplot(
            data = df,
            x = x1,
            y = x2,
            hue = hue,
            marker = '+',
            palette = colors,
            ax = ax)
    
    ax1.set_xlabel(x1)
    ax1.set_ylabel(x2)
    ax2.set_xlabel(x1)
    ax2.set_ylabel(x2)

    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    ax1.spines[['top', 'right']].set_visible(False)
    ax2.spines[['top', 'right']].set_visible(False)
    # Plotting the lines
    for i in np.arange(1.0, 0, -0.1):
        ax1.plot([0, i], [i, 0])
    # Title
    plt.suptitle(title)
    fig.tight_layout()
    #ax1.set_facecolor('lightgray')
    #ax2.set_facecolor('lightgray')

    fig.savefig("./October_5area_plot.png") # added by Mende
    # return fig, ax1, ax2
    del fig, ax1, ax2
    gc.collect()

def plot_mean_df(df, x1, x2, colors, hue, title) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, num=1, clear=True, figsize=(16, 6))
    #levels = df[hue].nunique()

    for ax in (ax1, ax2):
        sns.kdeplot(
            data = df,
            x = x1,
            y = x2,
            hue = hue,
            palette = colors,
            fill = False,
            multiple = 'layer',
            common_norm = False,
            ax = ax,
            levels = 1,
            #thresh = 0.005
        )

    ax1.set_xlabel(x1)
    ax1.set_ylabel(x2)
    ax2.set_xlabel(x1)
    ax2.set_ylabel(x2)

    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    ax1.spines[['top', 'right']].set_visible(False)
    ax2.spines[['top', 'right']].set_visible(False)

    for i in np.arange(1.0, 0, -0.1):
        ax1.plot([0, i], [i, 0])

    fig.tight_layout()
    plt.suptitle(title)

    plt.show()
    # plt.savefig("./October_5area_plot.png") # added by Mende
    del fig, ax1, ax2
    gc.collect()
    
def get_area_df(df_path = "./dji_msavi_gabor_intersection_fixed_area_df.parquet"):
    area_df = pd.read_parquet(df_path).drop_duplicates(subset = ['VegetationIndex', 'drone', 'class', 'patch_number'])
    area_df = area_df[area_df['drone'] != 'Altum'].reset_index(drop = True)

    area_df.loc[area_df['LocalHistEq_Gabor_Mask'], 'MSAVI_OG_Gabor_Intersection_Sum'] = area_df[area_df['LocalHistEq_Gabor_Mask']]['MSAVI_Local_Hist_Eq_Gabor_Intersection_Sum']

    area_df['Normalized_VI_Area'] = area_df['VI_Area']/area_df['Patch_Area']
    area_df['Normalized_Pure_GaborArea'] = area_df['Pure_GaborArea']/area_df['Patch_Area']
    area_df['Normalized_Unknown_Area'] = area_df['Unknown_Area']/area_df['Patch_Area']

    return area_df