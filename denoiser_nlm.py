import cv2
import numpy as np
from tqdm import tqdm
import logging

class Denoiser:
    def __init__(self, patch_size, h, num_balls_per_direction, pattern_size):
        self.patch_size = patch_size
        self.h = h
        self.num_balls_per_direction = num_balls_per_direction
        self.pattern_size = pattern_size

    def apply_nl_means(self, image, patch_size, h, num_balls_per_direction, pattern_size):
        if patch_size % 2 == 0:
            raise ValueError("Patch size of {} isn't odd".format(patch_size))
        patch_radius = int((patch_size - 1) / 2)

        # get all patches to evaluate
        logging.info("Collecting all balls")
        # balls, pixels_center = get_all_balls_image(image, patch_radius)

        image_processed = np.zeros_like(image)
        for index_y in tqdm(range(patch_radius, image.shape[0] - patch_radius)):
            for index_x in range(patch_radius, image.shape[1] - patch_radius):
                coordinates = np.asarray([index_y, index_x])

                # obtain balls around pixel
                balls, pixels_center = self.get_balls_neighborhood(image, coordinates, patch_radius,
                                                              num_balls_per_direction, pattern_size)

                mean_pixel = self.get_mean_for_pixel(image, balls, pixels_center, coordinates, patch_radius, h)

                image_processed[coordinates[0], coordinates[1]] = mean_pixel

        return image_processed

    def get_patch_around_pixel(self, image, coordinates, patch_radius):
        # coordinates_min = coordinates - patch_radius
        # coordinates_max = coordinates + patch_radius + 1  # +1 because of np slicing

        return image[coordinates[0] - patch_radius:coordinates[0] + patch_radius + 1,
                     coordinates[1] - patch_radius:coordinates[1] + patch_radius + 1]

    def get_ball_around_pixel(self, image, coordinates, patch_radius):
        patch = self.get_patch_around_pixel(image, coordinates, patch_radius)
        num_pixels_spatial = (patch_radius + 1) ** 2

        return np.divide(patch, num_pixels_spatial) # todo: * sum only for testing!

    def get_mean_for_pixel(self, image, balls, pixels_center, coordinates, patch_radius, h):

        if len(image.shape) == 2:
            num_colors = 1
        else:
            num_colors = image.shape[2]

        ball_pixel = self.get_ball_around_pixel(image, coordinates, patch_radius)

        # replicate ball to array like balls
        num_balls = balls.shape[0]
        ball_pixel_expanded = np.expand_dims(ball_pixel, axis=0)
        if num_colors > 1:
            ball_pixel_expanded = np.pad(ball_pixel_expanded, ((0, num_balls - 1), (0, 0), (0, 0), (0, 0)), mode='edge')
        else:
            ball_pixel_expanded = np.pad(ball_pixel_expanded, ((0, num_balls - 1), (0, 0), (0, 0)), mode='edge')

        # calculate distances. sum the squared pixel-wise differences for each ball along spatial axes 1 and 2.
        differences = np.subtract(balls, ball_pixel_expanded)
        distances = np.sum(differences ** 2, axis=(1, 2))

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

    def get_balls_neighborhood(self, image, coordinates_center, patch_radius, num_balls_per_direction, pattern_size):
        """

        :param image:
        :param coordinates_center:
        :param patch_radius:
        :param num_balls_per_direction:
        :param pattern_size: np array, size of repeating pattern for each axis
        :return:
        """
        list_balls = []
        list_center_pixels = []

        patch_size = patch_radius * 2 + 1

        shape_ball = (patch_size, patch_size)

        # obtain indices for y and x direction
        coordinates_min = coordinates_center - pattern_size * num_balls_per_direction
        coordinates_max = coordinates_center + pattern_size * num_balls_per_direction + 1

        coordinates_balls_y = np.arange(coordinates_min[0], coordinates_max[0], pattern_size[0])
        coordinates_balls_x = np.arange(coordinates_min[1], coordinates_max[1], pattern_size[1])

        # # for boundary evaluation
        # max_length_y = num_balls_per_direction * pattern_size[0] + patch_radius
        # max_length_x = num_balls_per_direction * pattern_size[1] + patch_radius

        for index_y in coordinates_balls_y:
            if index_y + patch_radius < image.shape[0] and index_y - patch_radius > 0:
                for index_x in coordinates_balls_x:
                    if index_x + patch_radius < image.shape[1] and index_x - patch_radius > 0:
                        coordinates_ball = np.asarray([index_y, index_x])
                        ball = self.get_ball_around_pixel(image, coordinates_ball, patch_radius)

                        list_balls.append(ball)
                        list_center_pixels.append(image[coordinates_ball[0],
                                                        coordinates_ball[1]])

        return np.asarray(list_balls), np.asarray(list_center_pixels)

    # def get_all_balls_image(self, image, patch_radius):
    #     """ Obtain array of shape (num_balls, patch_radius, patch_radius, num_colors) for balls,
    #         and array of shape (num_balls, num_colors) of center pixels
    #
    #     :param image:
    #     :param patch_radius:
    #     :return:
    #     """
    #     list_balls = []
    #     list_center_pixels = []
    #     nth_ball = 3
    #     # todo: introduce step-size to only get the nth balls
    #     for index_y in range(patch_radius, image.shape[0] - patch_radius, nth_ball):
    #         for index_x in range(patch_radius, image.shape[1] - patch_radius, nth_ball):
    #             coordinates_ball = np.asarray([index_y, index_x])
    #             ball = self.get_ball_around_pixel(image, coordinates_ball, patch_radius)
    #
    #             list_balls.append(ball)
    #             list_center_pixels.append(image[coordinates_ball[0],
    #                                             coordinates_ball[1]])
    #
    #     return np.asarray(list_balls), np.asarray(list_center_pixels)