import cv2
import numpy as np
import rawpy
from denoiser_nlm import Denoiser

# settings
num_cores = 2  # cores for multiprocessing
patch_radius = 4  # radius of patches. patch width is 2*radius+1
h = 1  # Scale parameter for exponential weight from patch distance. Higher value corresponds to smoother image. 1.1 seems good
rotate_patches = True  # not used

# Number of patches to search around each pixel in each direction (l, r, u, d).
# number of patches generated will be (2 * num_balls_per_direction)**2
num_balls_per_direction = 10

# slice to define region of interest to process. for quick experimenting
slice_width = 300
slice_center_x = 1500
slice_center_y = 3000

x_min = int (slice_center_x - slice_width/2)
x_max = int (slice_center_x + slice_width/2)
y_min = int (slice_center_y - slice_width/2)
y_max = int (slice_center_y + slice_width/2)

slice_denoise = np.s_[y_min:y_max, x_min:x_max]
slice_denoise = None  #  comment this line to use slice instead of filtering of whole image

# paths to input raw image and desired output
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/nikon_2.NEF"
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Fuji_5.RAF"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Fuji/Gruen.RAF"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Olympus_1600.ORF"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/denoised_images/Fuji_5_full_pd{}_pr{}_h{}.png".format(num_balls_per_direction,
                                                                                                                                 patch_radius,
                                                                                                                                 h)
# path to the camera profile generated with profile_camera.py
path_profile = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/results/profile_T10_200.csv"
# path_profile = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/results/profile_D40_1600.csv"

# read raw image
image_raw = rawpy.imread(path_input)

# perform nl means
denoiser = Denoiser(patch_radius, h, num_balls_per_direction, path_profile_camera=path_profile, num_cores=num_cores,
                    rotate_patches=rotate_patches)
denoiser.filter_image(image_raw, slice_denoise)

# demosaic and save image
image_processed = image_raw.postprocess(output_bps=16, use_auto_wb=True,
                                        # user_wb=(1,1,1,1),
                                        no_auto_bright=True,
                                        no_auto_scale=False,
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AMAZE)
cv2.imwrite(path_output, np.flip(image_processed, axis=2))