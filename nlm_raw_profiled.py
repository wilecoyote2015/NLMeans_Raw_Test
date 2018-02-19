import cv2
import numpy as np
from tqdm import tqdm
import logging
import rawpy
from denoiser_nlm import Denoiser

# settings
num_cores = 4
patch_radius = 4
# h = 0.000005  # for Nikon 2
# h = 0.00008  # for Nikon 1
# h = 0.000008  # for Fuji
h = 0.2  # for Fuji Dpreview
num_balls_per_direction = 7


slice_width = 100
slice_center_x = 1500
slice_center_y = 1000

x_min = int (slice_center_x - slice_width/2)
x_max = int (slice_center_x + slice_width/2)

y_min = int (slice_center_y - slice_width/2)
y_max = int (slice_center_y + slice_width/2)

slice_denoise = np.s_[y_min:y_max, x_min:x_max]

# import image
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/nikon_2.NEF"
# path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_nikon_1.png"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Fuji_2.RAF"
# path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_fuji_2.png"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Nikon_1_Profiled.RAF"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Nikon_2_profiled_test.png"
path_profile = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/results/profile_D40.csv"

image_raw = rawpy.imread(path_input)

# perform nl means
denoiser = Denoiser(patch_radius, h, num_balls_per_direction, path_profile_camera=path_profile, num_cores=num_cores)
denoiser.filter_image(image_raw, slice_denoise)

image_processed = image_raw.postprocess(output_bps=8, use_auto_wb=True)

cv2.imwrite(path_output, np.flip(image_processed, axis=2))