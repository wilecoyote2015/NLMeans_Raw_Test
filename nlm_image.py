import cv2
import numpy as np
from tqdm import tqdm
import logging
from denoiser_nlm import Denoiser
import cProfile

# settings
patch_size = 7
h = 10
num_balls_per_direction = 5
pattern_size = np.asarray((4,4))

# import image
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/crop_mini.png"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_image.png"

image = cv2.imread(path_input)

# perform nl means
denoiser = Denoiser(pattern_size, h, num_balls_per_direction, pattern_size)
image_processed = denoiser.apply_nl_means(image, patch_size, h, num_balls_per_direction, pattern_size)

cv2.imwrite(path_output, image_processed)