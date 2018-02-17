import cv2
import numpy as np
from tqdm import tqdm
import logging

def apply_nl_means(image, patch_size, h):
    if patch_size % 2 == 0:
        raise ValueError("Patch size of {} isn't odd".format(patch_size))
    patch_radius = int((patch_size - 1) / 2)

    # get all patches to evaluate
    logging.info("Collecting all balls")
    balls, pixels_center = get_all_balls_image(image, patch_radius)

    image_processed = np.zeros_like(image)
    for index_y in tqdm(range(patch_radius, image.shape[0] - patch_radius)):
        for index_x in range(patch_radius, image.shape[1] - patch_radius):
            coordinates = np.asarray([index_y, index_x])
            mean_pixel = get_mean_for_pixel(image, balls, pixels_center, coordinates, patch_radius, h)

            image_processed[coordinates[0], coordinates[1]] = mean_pixel

    return image_processed

def get_all_balls_image(image, patch_radius):
    """ Obtain array of shape (num_balls, patch_radius, patch_radius, num_colors) for balls,
        and array of shape (num_balls, num_colors) of center pixels

    :param image:
    :param patch_radius:
    :return:
    """
    list_balls = []
    list_center_pixels = []
    nth_ball = 3
    # todo: introduce step-size to only get the nth balls
    for index_y in range(patch_radius, image.shape[0] - patch_radius, nth_ball):
        for index_x in range(patch_radius, image.shape[1] - patch_radius, nth_ball):
            coordinates_ball = np.asarray([index_y, index_x])
            ball = get_ball_around_pixel(image, coordinates_ball, patch_radius)

            list_balls.append(ball)
            list_center_pixels.append(image[coordinates_ball[0],
                                            coordinates_ball[1]])

    return np.asarray(list_balls), np.asarray(list_center_pixels)

def get_mean_for_pixel(image, balls, pixels_center, coordinates, patch_radius, h):
    ball_pixel = get_ball_around_pixel(image, coordinates, patch_radius)

    # replicate ball to array like balls
    num_balls = balls.shape[0]
    ball_pixel_expanded = np.expand_dims(ball_pixel, axis=0)
    ball_pixel_expanded = np.pad(ball_pixel_expanded, ((0, num_balls - 1), (0,0), (0,0), (0,0)), mode='edge')

    # calculate distances. sum the squared pixel-wise differences for each ball along spatial axes 1 and 2.
    differences = np.subtract(balls, ball_pixel_expanded)
    distances = np.sum(differences**2, axis=(1,2))

    # weighths with exponential decay
    exponent = - np.divide(distances, h)
    weigths = np.exp(exponent)

    # sums of weigths for normalization. result is sum for each color channel
    sums_weights = np.sum(weigths, axis=0)

    # weighted summation of ball center pixels
    means = pixels_center * weigths
    mean = np.sum(means, axis=0)  # is mean for each color

    # normalization for weigths
    mean /= sums_weights

    return mean


    # sum_weights = 0.
    # value_filtered = 0.
    # for index_y in range(patch_radius, image.shape[0] - patch_radius):
    #     for index_x in range(patch_radius, image.shape[1] - patch_radius):
    #         coordinates_ball_other = np.asarray([index_y, index_x])
    #         ball_other = get_ball_around_pixel(image, coordinates_ball_other, patch_radius)
    #
    #         difference = ball_pixel - ball_other
    #
    #         # distance has length of numer of channels, as each color channel is filtered separately
    #         distance = np.sum(difference**2, axis=(0,1))
    #
    #         weight = np.exp(- distance/h)
    #         sum_weights += weight
    #
    #         value_filtered += weight * image[coordinates_ball_other[0],
    #                                          coordinates_ball_other[1]]
    #
    # # normalize for weights
    # value_filtered /= sum_weights
    #
    # return value_filtered


def get_patch_around_pixel(image, coordinates, patch_radius):

    coordinates_min = coordinates - patch_radius
    coordinates_max = coordinates + patch_radius + 1  # +1 because of np slicing

    return image[coordinates_min[0]:coordinates_max[0],
                 coordinates_min[1]:coordinates_max[1]]

def get_ball_around_pixel(image, coordinates, patch_radius):
    patch = get_patch_around_pixel(image, coordinates, patch_radius)
    num_pixels_spatial = patch[:,:,0].size

    return patch / num_pixels_spatial

# settings
patch_size = 7
h = 1

# import image
path_input = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/crop_mini.png"
path_output = "/run/media/bjoern/daten/Programming/Raw_NLM_Denoise/images/result_image.png"

image = cv2.imread(path_input)

# perform nl means
image_processed = apply_nl_means(image, patch_size, h)

cv2.imwrite(path_output, image_processed)