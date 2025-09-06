2025.06.16
In an effort to clean up this repository to make it better for whom it may concern, here's my explanation of the folders for better structure:

- `2025-05-17_human_classification_20_imgs`: So this folder name is quite self-explanatory. I chose 20 or so images, aligned them (the folder contains the scripts for this as align_pipeline). I used Jupyter notebooks to see the images and their respective MSAVI in random order and I gained some intuition about which features in given images most likely determines the image class of Best to Worst. I also tried some KMeans clustering algorithm here and connected components algorithms here as well.

- `2025-06-11_computation_on_all_images`: I ran a KMeans algorithm on all of the images as well as Moran's spatial coefficient.

- `combine_all_patches`: In the early phase of my diploma work(March 2025), I needed to recombine all the small patches of the orthomosaic into a one big image for better overall picture.

- `CSV_DJI_P4_2024`: There is the csv results of the temuujin's algorithm on 2024 data.

- `deeplearning`: So I tried some NN based models for classification using FastAI. Trained on october data and failed on july data.

- `Drone_Orthomosaics`: Generated orthos from WebODM as well as Metashape Pro.

- `GeoWithJulia`: I initially set this folder to be very heavily used but didn't. I mostly used Python throughout the semester. This folder contains pretty useful Pluto notebook called `notebook.jl` which loads in a ortho raster using Rasters.jl and performs some initial analysis.

- `image_per_land_explore`: So this is my exploration folder during the late phases of my diploma work. More specifically I tried edge detection, feature extraction using ViTs and more. ViTs suffered from large batches problem. Initial per land files were copied from `reflectance_msavi_threshold` folder.

- `processed_*`: These are folders appears to contain non-relevant info

- `PySegment`: This is a tutorial for medical image segmentation using `monai` library.

- `RandomForests`: So this folder contains much much more than the name implies. I last modified this folder on May 2nd 2025. What I can tell is that I first wrote a Python script to align images from each land type. Then I tried to segment out the ground from vegetation using thresholding method. To establish a reliable threshold I used image labeler in `napari` which admittedly was pretty rudimentary. I also attempted Object Based Image Analysis and some micasense tutorial as well. In this folder I established thresholding for ground segmentation. the `Bad` land images were pretty low in MSAVI values for some reason.

- `reflectance_msavi_threshold` is a folder which heavily based on `RandomForests` folder. I realized we need to normalize the images using the sun sensor of the DJI drone. However, as there were no commercially available or well established library for drone image normalization, I resorted to a rudimentary GitHub repo with almost no recongition called p4m. It's based on the micasense library. This achieved an MSAVI plots with more variability. The issue of `Bad` land image has been dampened but still had visually different look.

- `Reports`: Long forgotten folder containing ODM report and DJI P4 Guide.

- `reseda`: I here used LandSat imagery to segment water and fields from Berlin images. The tutorial was adapted from R. Advice was from Batnyambuu to take a look at this tutorial.

- `Results_Altum_*`: used temuujin's algo on Altum 2024 images. Didn't bother analyzing the results.

- `shapefiles`: SE

- `testing`: Maybe testing Temuujin algo???

- `Visualization`: Make plots from Temuujin algo results

Lastly as a reminder, the temuujin's initial algorithm is in following Python scripts. 

- `gabor_texture.py`
- `plot_utils.py`
- `cli_gabor_segmentation_of_tiff_indir.py`: cli interface


