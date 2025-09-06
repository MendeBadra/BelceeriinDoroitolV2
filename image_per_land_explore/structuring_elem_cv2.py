import cv2
import numpy as np
import imageio.v3 as iio

# Functions
def overlay_msavi_labels(msavi, labels, num_labels)->np.ndarray:
    """
    Overlay blob contours on the MSAVI image.

    Args:
        msavi (np.ndarray): MSAVI image, float in [0,1].
        labels (np.ndarray): Labeled image from connectedComponentsWithStats.
        num_labels (int): Number of labels.

    Returns:
        np.ndarray: Overlay image with contours.
    """
    overlay = cv2.cvtColor((msavi  * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for label in range(1, num_labels):  # Skip label 0 (background)
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)  # Green outlines
    return overlay

def find_opened(veg_mask):
    """
    find the opened mask

    Args:
        veg_mask (np.ndarray): Vegetation mask after thresholding. Veg section must be truthy.

    Returns:
        np.ndarray: opened (don't know what it is doing)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(veg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened


# then use cv2.connectedComponentsWithStats

if __name__ == "__main__":
    # Now trying on the bad land image
    threshold_value = 0.1994333333
    msavi = iio.imread("2025-05-21_working_imgs/DJI_002INDS.TIF")[:,:,0]
