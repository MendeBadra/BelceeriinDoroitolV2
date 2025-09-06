import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    2025.05.21
    # Trying out different images processing methods to compute the following
    For each blob in the image compute its:
    - Area
    - Density
    - Intensity of the MSAVI

    These are basically meaning: if an area of a blob is very big (талбай нь их бол) -> bad 
    if density of a blob is high -> weed
    if intensity of the msavi showing high values compared to it's surroundings -> weed
    """
    )
    return


@app.cell
def _():
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from skimage.exposure import rescale_intensity
    from scipy.ndimage import gaussian_filter
    from scipy import ndimage

    import marimo as mo

    import imageio.v3 as iio
    from pathlib import Path
    return (
        ListedColormap,
        cv2,
        gaussian_filter,
        iio,
        mo,
        ndimage,
        np,
        pd,
        plt,
        rescale_intensity,
    )


@app.cell
def _(ListedColormap, np, plt, rescale_intensity):
    # Functions
    def plot_index(index, dir_name,index_name="MSAVI", cmap="jet", threshold=-1.0, norm_index=False, return_mask=False):
        fig, ax = plt.subplots(figsize=(6, 5))
        ground_mask = index < threshold
        vmin, vmax = np.nanmin(index), np.nanmax(index)
        im = ax.imshow(index if not norm_index else rescale_intensity(index, in_range=(vmin, vmax), out_range=(-1, 1)),
                        cmap=cmap, vmin=-1, vmax=1)
        ax.axis("off")
        # Create custom colormap
        # 0 (False) will be transparent, 1 (True) will be black
        # RGBA format: (Red, Green, Blue, Alpha)
        # Transparent: (0, 0, 0, 0) - any color with alpha 0
        # Black: (0, 0, 0, 1)
        mask_cmap = ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])

        # Overlay the ground_mask
        # Convert boolean mask to int (False->0, True->1)
        # Ensure the mask is plotted on top, and its values map correctly to the new colormap
        ax.imshow(ground_mask.astype(int), cmap=mask_cmap, vmin=0, vmax=1)

        fig.colorbar(im, ax=ax, label=index_name)
        plt.title(f"{index_name} of {dir_name}")
        plt.tight_layout()
        if return_mask == True:
            return ground_mask, plt.gca()
        else:
            return plt.gca()

    def show_image(image, dir_name, ground_mask=None, cmap="viridis"):
        plt.figure(figsize=(6, 5))
        if (ground_mask is not None):
            if len(image.shape) == 3:
                ground_mask = np.expand_dims(ground_mask, axis=-1)
            image = image * ~ground_mask

        if len(image.shape) < 3:
            plt.imshow(image, cmap=cmap)
        elif len(image.shape) == 3:
            plt.imshow(image)
        plt.title(f"Image of {dir_name}")
        plt.axis('off')
        plt.tight_layout()
        return plt.gca()
    return plot_index, show_image


@app.cell
def _(pd):
    df = pd.read_csv("thresholds.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now let's import the `Bad` and `Worst` images and try thresholding methods.""")
    return


@app.cell
def _(iio):
    worst_rgb = iio.imread("2025-05-21_working_imgs/DJI_0020_124fplan.JPG")
    bad_rgb = iio.imread("2025-05-21_working_imgs/DJI_0530_106fplan.JPG")

    worst_indices = iio.imread("2025-05-21_working_imgs/DJI_002INDS.TIF")
    bad_indices = iio.imread("2025-05-21_working_imgs/DJI_053INDS.TIF")
    worst_msavi = worst_indices[:,:,0]
    bad_msavi = bad_indices[:,:,0]
    return bad_msavi, bad_rgb, worst_msavi, worst_rgb


@app.cell
def _(df):
    bad_thresh = df[df["land_type"] == "bad"]["threshold"].values[0]
    return (bad_thresh,)


@app.cell
def _(bad_rgb, show_image):
    show_image(bad_rgb, "bad")
    return


@app.cell
def _(bad_msavi, bad_thresh, plot_index):
    bad_ground_mask, axis = plot_index(bad_msavi, index_name="MSAVI", dir_name="bad", cmap="RdYlGn", threshold=bad_thresh, return_mask=True)
    axis
    return (bad_ground_mask,)


@app.cell
def _():
    return


@app.cell
def _(bad_ground_mask, bad_rgb, show_image):
    show_image(bad_rgb, "bad", ground_mask=bad_ground_mask)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Looks like thresholding the MSAVI from above may yield good results.""")
    return


@app.cell
def _():
    # upper_threshold = mo.ui.slider(bad_thresh, .9, 0.01)
    # upper_threshold
    return


@app.cell
def _():
    class Threshold:
        def __init__(self, val):
            self.value = val

    upper_threshold = Threshold(0.68)

    # upper_threshold = 0.68
    return (upper_threshold,)


@app.cell
def _(bad_msavi, bad_rgb, np, plt, upper_threshold):
    high_intensity_section = bad_msavi > upper_threshold.value
    high_intensity_section = np.expand_dims(high_intensity_section, axis=-1)
    plt.imshow(high_intensity_section * bad_rgb, cmap="gray")
    return (high_intensity_section,)


@app.cell
def _(high_intensity_section, plt):
    plt.imshow(high_intensity_section, cmap="gray")
    # plt.axis('off')
    # plt.gca()
    return


@app.cell
def _(high_intensity_section, plt):
    from skimage.morphology import binary_closing, binary_opening, disk

    mask_clean = binary_closing(high_intensity_section.squeeze(), disk(5))
    mask_clean = binary_opening(mask_clean, disk(3))
    plt.imshow(mask_clean, cmap="gray")
    return binary_closing, binary_opening, disk, mask_clean


@app.cell
def _(mo):
    mo.md(r"""### Trying watershed""")
    return


@app.cell
def _(mask_clean, np, plt):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    distance = ndi.distance_transform_edt(mask_clean)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask_clean)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=mask_clean)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(mask_clean, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.gca()
    return (ndi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Trying connected-components""")
    return


@app.cell
def _(mask_clean, ndimage, plt):
    bad_labeled_array, bad_num_features = ndimage.label(mask_clean)
    plt.imshow(bad_labeled_array, cmap='nipy_spectral')
    plt.title(f"{bad_num_features} blobs found")
    plt.gca()
    return


@app.cell
def _(mask_clean):
    from skimage.measure import label, regionprops
    bad_labels = label(mask_clean)
    bad_regions = regionprops(bad_labels)
    bad_regions
    return bad_labels, bad_regions


@app.cell
def _(bad_labels, np):
    np.unique(bad_labels)
    return


@app.cell
def _(bad_regions, plt):
    plt.imshow(bad_regions[1].image)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It's easy to see from here that `connected_component` method is far better for the `bad` land""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Now let's take a look at `worst`""")
    return


@app.cell
def _(show_image, worst_rgb):
    show_image(worst_rgb, dir_name="worst")
    return


@app.cell
def _(df, plot_index, worst_msavi):
    worst_thresh = df[df["land_type"] == "worst"]["threshold"].values[0]
    worst_ground_mask, worst_fig = plot_index(worst_msavi, "MSAVI", "worst", cmap="RdYlGn", threshold=worst_thresh, return_mask=True)
    worst_fig
    return worst_ground_mask, worst_thresh


@app.cell
def _(show_image, worst_ground_mask, worst_rgb):
    show_image(worst_rgb, dir_name="worst", ground_mask=worst_ground_mask)
    return


@app.cell
def _(worst_thresh):
    print(worst_thresh)
    return


@app.cell
def _(mo):
    worst_upper_thresh = mo.ui.slider(0.0, 0.9, 0.01)
    worst_upper_thresh
    return (worst_upper_thresh,)


@app.cell
def _(show_image, variable_worst_mask, worst_rgb):
    show_image(worst_rgb, dir_name="worst", ground_mask=~variable_worst_mask)
    return


@app.cell
def _(plt, worst_msavi, worst_upper_thresh):
    variable_worst_mask = worst_msavi > worst_upper_thresh.value
    plt.imshow(variable_worst_mask, cmap="gray")
    return (variable_worst_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## This section is experiment

          Let's try thresholding with extremely blurred version of the MSAVI
    """
    )
    return


@app.cell
def _(cv2, plot_index, worst_msavi):
    worst_msavi_blurred = cv2.GaussianBlur(worst_msavi, (101, 101), 0, borderType=cv2.BORDER_REPLICATE)
    plot_index(worst_msavi_blurred, "worst", cmap="RdYlGn")
    return (worst_msavi_blurred,)


@app.cell
def _(plt, worst_msavi, worst_msavi_blurred):
    plt.imshow(worst_msavi > worst_msavi_blurred, cmap="gray")
    return


@app.cell
def _(bad_msavi, gaussian_filter, plot_index):
    bad_msavi_blurred = gaussian_filter(bad_msavi, sigma=10, mode='reflect')
    plot_index(bad_msavi_blurred, "bad", cmap="RdYlGn")
    return (bad_msavi_blurred,)


@app.cell
def _(bad_msavi, bad_msavi_blurred, plt):

    plt.imshow(bad_msavi > bad_msavi_blurred, cmap="gray")
    return


@app.cell
def _(mo):
    mo.md("""## Back to the main course...""")
    return


@app.cell
def _(binary_closing, binary_opening, disk, plt, variable_worst_mask):
    # Forgot the morphology operations on worst
    worst_mask_clean = binary_closing(variable_worst_mask, disk(2))
    worst_mask_clean = binary_opening(worst_mask_clean, disk(2))
    plt.imshow(worst_mask_clean, cmap="gray")
    return (worst_mask_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This didn't really create the thing that I wanted. How can I detect blobs from this image?""")
    return


@app.cell
def _(ndimage, plt, variable_worst_mask):
    # Let's for a moment forget about these gaussians

    # Now turn to detecting these blobs in worst image
    labeled_array, num_features = ndimage.label(variable_worst_mask)
    plt.imshow(labeled_array, cmap='nipy_spectral')
    plt.title(f"{num_features} blobs found")
    plt.show()
    return


@app.cell
def _(ndimage, plt, worst_mask_clean):
    def _():
        labeled_array, num_features = ndimage.label(worst_mask_clean)
        plt.imshow(labeled_array, cmap='nipy_spectral')
        plt.title(f"{num_features} blobs found")
        return plt.show()


    _()
    return


@app.cell
def _(ndimage, plt, variable_worst_mask, worst_rgb):
    # from scipy import ndimage
    # import matplotlib.pyplot as plt
    from skimage.color import rgb2gray
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 5))

    worst_laplace = ndimage.laplace(variable_worst_mask)

    _ax1.imshow(worst_rgb)
    _ax1.set_title("Original Image")

    _ax2.imshow(worst_laplace, cmap='gray')
    _ax2.set_title("Laplace Filtered Image")

    plt.gca()
    return rgb2gray, worst_laplace


@app.cell
def _(plot_index, worst_msavi, worst_thresh):
    plot_index(worst_msavi, dir_name="Worst", cmap="viridis", threshold=worst_thresh)
    return


@app.cell
def _(show_image, worst_laplace):
    show_image(worst_laplace, "dir_name", cmap="gray")
    return


@app.cell
def _(ndi, np, plt, rgb2gray, worst_rgb):
    def plot_sobel(gray_img):
        kernel_horizontal = np.asarray([-1, 0, 1]).reshape((1, 3))
        kernel_vertical = np.asarray([-1, 0, 1]).reshape((3, 1))

        worst_im_horizontal = ndi.convolve(gray_img, kernel_horizontal)
        worst_im_vertical = ndi.convolve(gray_img, kernel_vertical)

        # Combine horizontal and vertical gradients to get the magnitude
        worst_sobel = np.sqrt(np.square(worst_im_horizontal) + np.square(worst_im_vertical))

        plt.imshow(worst_sobel, cmap='gray')
        plt.title("Sobel Filtered Image")
        return plt.gca()

    plot_sobel(rgb2gray(worst_rgb))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2. Computing the area for each blob""")
    return


@app.cell
def _(mask_clean, ndimage, np, plt):
    def _():
        # Let's start by computing the area of each segment:
        _labeled_array, num_features = ndimage.label(mask_clean)
        _areas = ndimage.sum(mask_clean, _labeled_array, index=np.arange(_labeled_array.max() + 1))

        plt.imshow(_labeled_array, cmap='nipy_spectral')
        plt.title(f"{num_features} blobs found")
        plt.colorbar(label="Blob ID")
        plt.xlabel("X")
        plt.ylabel("Y")
        return plt.gca()


    _()
    return


@app.cell
def _():
    import altair as alt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Interactive Labeled Segment Analysis

    Hover over the segments in the image below to see their properties.
    The image displays the blobs found by the labeling process.
    """
    )
    return


@app.cell
def _(plt):
    import skimage as ski
    coins = ski.data.coins()
    plt.imshow(coins, cmap="gray")
    return coins, ski


@app.cell
def _(coins, ski):
    threshold_value = ski.filters.threshold_otsu(coins)
    print(threshold_value)
    return


@app.cell
def _(coins, plt, ski):
    coins_edges = ski.feature.canny(coins / 255.)
    plt.imshow(coins_edges)
    return (coins_edges,)


@app.cell
def _(coins_edges, plt):
    import scipy as sp
    fill_coins = sp.ndimage.binary_fill_holes(coins_edges)
    plt.imshow(fill_coins)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Edge-based segmentation""")
    return


@app.cell
def _(bad_msavi, plt):
    plt.imshow(bad_msavi)
    return


@app.cell
def _(bad_msavi, ski):
    edges = ski.feature.canny(bad_msavi)
    return (edges,)


@app.cell
def _(edges, plt):
    plt.imshow(edges)
    return


@app.cell
def _():
    return


@app.cell
def _(bad_ground_mask, plt):
    from skimage import measure

    # `binary_mask` is your thresholded image (values: 0 and 1)
    contours = measure.find_contours(bad_ground_mask, level=0.5)
    def _():
        # Plot contours
        fig, ax = plt.subplots()
        ax.imshow(bad_ground_mask, cmap='gray')
    
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
    
        plt.title("Boundaries of blobs")
        return plt.gca()
    _()
    return


@app.cell
def _(np, worst_rgb):
    worst_red, worst_green, worst_blue = np.permute_dims(worst_rgb, axes=(2,0,1))
    return worst_blue, worst_green, worst_red


@app.cell
def _(plt, worst_msavi):
    plt.imshow(worst_msavi , cmap="viridis")
    return


@app.cell
def _(plt, worst_blue, worst_green, worst_red):
    worst_ngi = 2*worst_green - worst_red - worst_blue
    plt.imshow(worst_ngi, cmap="gray", vmin=0, vmax=256)
    return


@app.cell
def _(worst_blue, worst_green, worst_red):
    threshold_g = 0

    green_mask = (worst_green > threshold_g) & (worst_green > worst_red) & (worst_green > worst_blue)
    return (green_mask,)


@app.cell
def _(green_mask, plt):
    plt.imshow(green_mask)
    return


@app.cell
def _(bad_ground_mask, bad_msavi_nonnan, plt):
    from sklearn.cluster import KMeans

    bad_msavi_masked = bad_msavi_nonnan * ~bad_ground_mask

    plt.imshow(bad_msavi_masked)
    return KMeans, bad_msavi_masked


@app.cell
def _(bad_msavi, np):
    bad_msavi_nonnan = np.nan_to_num(bad_msavi, nan=0.0)
    return (bad_msavi_nonnan,)


@app.cell
def _(mo):
    mo.md(
        r"""
    2025.05.23

    # KMeans idea
    using kmeans in Sklearn
    """
    )
    return


@app.cell
def _(k, plt):
    def draw_masks(weed, grass, ground):
        # Assuming weed, grass, and ground are 2D arrays (same shape)
        fig, ax = plt.subplots(1, 3, figsize=(150, 50))  # 1 row, 3 columns
    
        ax[0].imshow(weed, cmap='gray')
        ax[0].set_title('Weed')
        ax[0].axis('off')
    
        ax[1].imshow(grass, cmap='gray')
        ax[1].set_title('Grass')
        ax[1].axis('off')
    
        ax[2].imshow(ground, cmap='gray')
        ax[2].set_title('Ground')
        ax[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    def draw_side_by_side(msavi_masked, segmented_image):
        # Display original and segmented image
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(msavi_masked, cmap='gray')
        axs[0].set_title('Original Grayscale Image')
        axs[0].axis('off')
    
        axs[1].imshow(segmented_image, cmap='viridis')
        axs[1].set_title(f'K-means Segmented (k={k})')
        axs[1].axis('off')
    
        plt.tight_layout()
        return plt.gca()

    return draw_masks, draw_side_by_side


@app.cell
def _(KMeans, bad_msavi_masked):
    h, w = bad_msavi_masked.shape
    pixels = bad_msavi_masked.reshape(-1, 1)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    # Map each pixel to its cluster center
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
    bad_segmented_image = segmented_pixels.reshape(h, w)

    #draw_side_by_side(bad_msavi_masked, bad_segmented_image)
    # draw_side_by_side(bad_msavi_masked, bad_segmented_image)
    return bad_segmented_image, k


@app.cell
def _(bad_msavi_masked, bad_segmented_image, draw_side_by_side):
    draw_side_by_side(bad_msavi_masked, bad_segmented_image)
    return


@app.cell
def _(bad_segmented_image, draw_masks, np):
    print(f"KMeans Unique: {np.unique(bad_segmented_image)}")

    # Segment 1: Ground (but kinda distorted)
    ground = np.where(bad_segmented_image < 0.2, bad_segmented_image, 0)

    # Segment 2: Grass
    grass = np.where((bad_segmented_image >= 0.2) & (bad_segmented_image < 0.6), bad_segmented_image, 0)
    # Segment 3: Weed
    weed = np.where(bad_segmented_image > 0.6, bad_segmented_image, 0)



    draw_masks(weed, grass, ground)
    return ground, weed


@app.cell
def _(bad_ground_mask, plt):
    plt.imshow(bad_ground_mask)
    return


@app.cell
def _(bad_ground_mask, ground, np):
    np.sum(ground != 0), np.sum(bad_ground_mask)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Trying it on worst""")
    return


@app.cell
def _(np, worst_ground_mask, worst_msavi):
    worst_msavi_nonnan = np.nan_to_num(worst_msavi, nan=0.0)
    worst_msavi_masked = worst_msavi_nonnan * ~worst_ground_mask
    return (worst_msavi_masked,)


@app.cell
def _(KMeans, draw_side_by_side, np, worst_msavi_masked):
    def worst_calculate():
        h, w = worst_msavi_masked.shape
        pixels = worst_msavi_masked.reshape(-1, 1)

        k = 3
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
        # Map each pixel to its cluster center
        segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
        print(f"KMeans labels: {kmeans.cluster_centers_}")
        print(f"KMeans unique pixel values {np.unique(segmented_pixels)}")
        segmented_image = segmented_pixels.reshape(h, w)
        ground_val, grass_val, weed_val = sorted(np.unique(segmented_pixels))
        ground = np.where(segmented_image == ground_val, segmented_image, 0)
        grass = np.where(segmented_image == grass_val, segmented_image, 0)
        weed = np.where(segmented_image == weed_val, segmented_image, 0)
    
        return ground, grass, weed, draw_side_by_side(worst_msavi_masked, segmented_image)

    worst_ground, worst_grass, worst_weed, _plot = worst_calculate()
    _plot
    return worst_grass, worst_ground, worst_weed


@app.cell
def _(draw_masks, worst_grass, worst_ground, worst_weed):
    # print(np.unique(worst_seg))


    draw_masks(worst_weed, worst_grass, worst_ground)
    return


@app.cell
def _(worst_weed):
    worst_weed
    return


@app.cell
def _(cv2, np):
    def morphological_transform(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        return opened
    
    return (morphological_transform,)


@app.cell
def _(morphological_transform, plt, worst_weed):
    worst_opened = morphological_transform(worst_weed != 0)
    plt.imshow(worst_opened)
    return


@app.cell
def _(plt, worst_rgb):
    plt.imshow(worst_rgb)
    return


@app.cell
def _(morphological_transform, plt, weed):
    bad_opened = morphological_transform(weed != 0)
    plt.imshow(bad_opened)
    return


@app.cell
def _(bad_rgb, plt):
    plt.imshow(bad_rgb)
    return


@app.cell
def _():
        
    return


if __name__ == "__main__":
    app.run()
