import cv2
import numpy as np
from tqdm import tqdm
import logging
from denoiser_nlm import Denoiser
import rawpy

# settings
num_cores = 1
patch_radius = 3
# h = 0.000005  # for Nikon 2
# h = 0.00008  # for Nikon 1
# h = 0.000008  # for Fuji
h = 0.0000005  # for Fuji Dpreview
num_balls_per_direction = 7


slice_width = 150
slice_center_x = 900
slice_center_y = 2700

x_min = int (slice_center_x - slice_width/2)
x_max = int (slice_center_x + slice_width/2)

y_min = int (slice_center_y - slice_width/2)
y_max = int (slice_center_y + slice_width/2)

slice_denoise = np.s_[y_min:y_max, x_min:x_max]

# import image
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/nikon_1.NEF"
# path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_nikon_1.png"
# path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Fuji_2.RAF"
# path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_fuji_2.png"
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/Fuji_Dpreview.RAF"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_fuji_Dpreview.png"

image_raw = rawpy.imread(path_input)

data_to_denoise = np.copy(image_raw.raw_image[slice_denoise]).astype(np.float32)
max = np.amax(image_raw.raw_image)

# remove negative values
data_to_denoise[np.logical_or(data_to_denoise < 0., np.isnan(data_to_denoise))] = 0.

# todo: for testing!!! scale with sqrt to approximate gaussian normalizaton
# data_to_denoise /= np.sqrt(data_to_denoise)

# normalize
data_to_denoise /= max  # todo: use this when not sqrt!
# data_to_denoise /= np.sqrt(max)

# set nans to 0
data_to_denoise[np.isnan(data_to_denoise)] = 0.

# perform nl means
pattern_size = np.asarray(image_raw.raw_pattern.shape)
denoiser = Denoiser(patch_radius, h, num_balls_per_direction, pattern_size, num_cores=num_cores)
data_processed = denoiser.apply_nl_means(data_to_denoise)

# rescale data
# data_processed *= np.sqrt(max) * np.sqrt(data_processed)

# todo: use this if without sqrt!
image_raw.raw_image[...][slice_denoise] = (data_processed * max).astype(np.uint16)
# image_raw.raw_image[...][slice_denoise] = (data_processed).astype(np.uint16)

image_processed = image_raw.postprocess(output_bps=8, use_auto_wb=True)

cv2.imwrite(path_output, np.flip(image_processed, axis=2))