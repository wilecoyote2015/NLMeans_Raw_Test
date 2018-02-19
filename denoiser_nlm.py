import cv2
import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import Pool
import logging
from csv import DictReader
from copy import deepcopy
from common_functions import ascombe_transform_scale, inverse_ascombe_transform_scale

class Denoiser:
    def __init__(self, patch_radius, h, num_balls_per_direction, pattern_size=None, path_profile_camera=None, num_cores=4):
        self.patch_radius = patch_radius
        self.h = h
        self.num_balls_per_direction = num_balls_per_direction
        self.pattern_size = pattern_size
        self.num_cores = num_cores

        if path_profile_camera is not None:
            self.parameters_camera = self.get_camera_parameters(path_profile_camera)
        else:
            self.parameters_camera = None

    def filter_image(self, image_raw, slice_denoise=None):
        # store old pattern size
        pattern_size_old = deepcopy(self.pattern_size)

        # set new pattern size
        self.pattern_size = np.asarray(image_raw.raw_pattern.shape)

        # copy the image
        # image_raw = deepcopy(image_raw) # todo: not possible...
        image_data = image_raw.raw_image

        # clean NANs
        image_data[np.isnan(image_data)] = 0.

        # transform the data to zero variance
        image_data_transformed = self.ascombe_transform_data(image_data.astype(np.float32),
                                                             image_raw)

        # perform nl means
        if slice_denoise is not None:
            data_to_filter = image_data_transformed[slice_denoise]
        else:
            data_to_filter = image_data_transformed

        image_data_filtered = self.apply_nl_means(data_to_filter)

        if slice_denoise is not None:
            image_data_transformed[slice_denoise] = image_data_filtered

        # re-transform image data
        image_data_filtered_backtransformed = self.ascombe_transform_data(image_data_transformed, image_raw, inverse=True)

        # write filtered data into image
        image_raw.raw_image[...] = image_data_filtered_backtransformed.astype(np.uint16)

        # reset patter size
        self.pattern_size = pattern_size_old

        return image_raw

    def ascombe_transform_data(self, image_data, image_raw, inverse=False):
        raw_pattern = image_raw.raw_pattern
        color_indices = image_raw.raw_colors
        image_data_transformed = np.zeros_like(image_data)
        for color_index in raw_pattern.flatten():
            alpha = self.parameters_camera[color_index]['alpha']
            beta = self.parameters_camera[color_index]['beta']
            pixels_color = [color_indices == color_index]
            if inverse:
                image_data_transformed[pixels_color] = inverse_ascombe_transform_scale(image_data[pixels_color],
                                                                               alpha,
                                                                               beta)
            else:
                image_data_transformed[pixels_color] = ascombe_transform_scale(image_data[pixels_color],
                                                                   alpha,
                                                                   beta)

        return image_data_transformed

    def apply_nl_means(self, image):
        # get all patches to evaluate
        logging.info("Collecting all balls")
        # balls, pixels_center = get_all_balls_image(image, patch_radius)

        if self.num_cores > 1:
            data = []
            for index_y in range(self.patch_radius, image.shape[0] - self.patch_radius):
                data.append({'image': image,
                             'index_y': index_y})

            images_results = self.map_function_with_tqdm_multiprocessing_to_list(self.apply_nl_means_row, data, 4)
            image_processed = np.sum(np.asarray(images_results), axis=0)
        else:
            image_processed = np.zeros_like(image)
            for index_y in tqdm(range(self.patch_radius, image.shape[0] - self.patch_radius)):
                for index_x in range(self.patch_radius, image.shape[1] - self.patch_radius):
                    coordinates_pixel = np.asarray([index_y, index_x])

                    #### obtain balls around pixel
                    list_balls = []
                    list_center_pixels = []

                    # obtain indices for y and x direction
                    coordinates_min = coordinates_pixel - self.pattern_size * self.num_balls_per_direction
                    coordinates_max = coordinates_pixel + self.pattern_size * self.num_balls_per_direction + 1

                    coordinates_balls_y = np.arange(coordinates_min[0], coordinates_max[0], self.pattern_size[0])
                    coordinates_balls_x = np.arange(coordinates_min[1], coordinates_max[1], self.pattern_size[1])

                    # delete coordinates that are out of bounds
                    good_elements_y = np.logical_and(coordinates_balls_y + self.patch_radius < image.shape[0],
                                                     coordinates_balls_y - self.patch_radius > 0)
                    good_elements_x = np.logical_and(coordinates_balls_x + self.patch_radius < image.shape[1],
                                                     coordinates_balls_x - self.patch_radius > 0)

                    # # for boundary evaluation
                    # max_length_y = num_balls_per_direction * pattern_size[0] + patch_radius
                    # max_length_x = num_balls_per_direction * pattern_size[1] + patch_radius

                    ### get all balls for all balls centers
                    for index_ball_center_y in coordinates_balls_y[good_elements_y]:
                        for index_ball_center_x in coordinates_balls_x[good_elements_x]:
                            ####  get ball around pixel
                            patch = image[index_ball_center_y - self.patch_radius:index_ball_center_y + self.patch_radius + 1,
                                    index_ball_center_x - self.patch_radius:index_ball_center_x + self.patch_radius + 1]
                            num_pixels_spatial = (self.patch_radius + 1) ** 2

                            ball = np.divide(patch, num_pixels_spatial)  # todo: * sum only for testing!

                            list_balls.append(ball)
                            list_center_pixels.append(image[index_ball_center_y,
                                                            index_ball_center_x])

                    balls, pixels_center =  np.asarray(list_balls), np.asarray(list_center_pixels)

                    mean_pixel = self.get_mean_for_pixel(image, balls, pixels_center, coordinates_pixel)

                    image_processed[coordinates_pixel[0], coordinates_pixel[1]] = mean_pixel

        return image_processed

    def apply_nl_means_row(self, data):
        image = data['image']
        index_y = data['index_y']
        image_processed = np.zeros_like(image)
        for index_x in range(self.patch_radius, image.shape[1] - self.patch_radius):
            coordinates_pixel = np.asarray([index_y, index_x])

            #### obtain balls around pixel
            list_balls = []
            list_center_pixels = []

            # obtain indices for y and x direction
            coordinates_min = coordinates_pixel - self.pattern_size * self.num_balls_per_direction
            coordinates_max = coordinates_pixel + self.pattern_size * self.num_balls_per_direction + 1

            coordinates_balls_y = np.arange(coordinates_min[0], coordinates_max[0], self.pattern_size[0])
            coordinates_balls_x = np.arange(coordinates_min[1], coordinates_max[1], self.pattern_size[1])

            # delete coordinates that are out of bounds
            good_elements_y = np.logical_and(coordinates_balls_y + self.patch_radius < image.shape[0],
                                             coordinates_balls_y - self.patch_radius > 0)
            good_elements_x = np.logical_and(coordinates_balls_x + self.patch_radius < image.shape[1],
                                             coordinates_balls_x - self.patch_radius > 0)

            # # for boundary evaluation
            # max_length_y = num_balls_per_direction * pattern_size[0] + patch_radius
            # max_length_x = num_balls_per_direction * pattern_size[1] + patch_radius

            ### get all balls for all balls centers
            for index_ball_center_y in coordinates_balls_y[good_elements_y]:
                for index_ball_center_x in coordinates_balls_x[good_elements_x]:
                    ####  get ball around pixel
                    patch = image[index_ball_center_y - self.patch_radius:index_ball_center_y + self.patch_radius + 1,
                            index_ball_center_x - self.patch_radius:index_ball_center_x + self.patch_radius + 1]
                    num_pixels_spatial = (self.patch_radius + 1) ** 2

                    ball = np.divide(patch, num_pixels_spatial)  # todo: * sum only for testing!

                    list_balls.append(ball)
                    list_center_pixels.append(image[index_ball_center_y,
                                                    index_ball_center_x])

            balls, pixels_center = np.asarray(list_balls), np.asarray(list_center_pixels)

            mean_pixel = self.get_mean_for_pixel(image, balls, pixels_center, coordinates_pixel)

            image_processed[coordinates_pixel[0], coordinates_pixel[1]] = mean_pixel

        return image_processed

    def map_function_with_tqdm_multiprocessing_to_list(self, function, data_to_process, num_cores):
        """ Process a function via multiprocessing and write the result into a list by merging the results returned by the
            function into a list


        :param function: a Function returning a dict
        :param data_to_process: list of dicts that are passed to the function
        :param num_cores:
        :return: dict, merged from resulting dicts of function
        """
        pool = Pool(processes=num_cores)
        result = []
        for result_iteration in tqdm(pool.imap_unordered(function, data_to_process),
                                     total=len(data_to_process)):
            result.append(result_iteration)

        return result

    def get_patch_around_pixel(self, image, coordinates):
        # coordinates_min = coordinates - patch_radius
        # coordinates_max = coordinates + patch_radius + 1  # +1 because of np slicing

        return image[coordinates[0] - self.patch_radius:coordinates[0] + self.patch_radius + 1,
                     coordinates[1] - self.patch_radius:coordinates[1] + self.patch_radius + 1]

    def get_ball_around_pixel(self, image, coordinates):
        patch = self.get_patch_around_pixel(image, coordinates)
        num_pixels_spatial = (self.patch_radius + 1) ** 2

        return np.divide(patch, num_pixels_spatial) # todo: * sum only for testing!

    def get_mean_for_pixel(self, image, balls, pixels_center, coordinates):

        if len(image.shape) == 2:
            num_colors = 1
        else:
            num_colors = image.shape[2]

        ball_pixel = self.get_ball_around_pixel(image, coordinates)

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
        exponent = - np.divide(distances, self.h)
        weigths = np.exp(exponent)

        # sums of weigths for normalization. result is sum for each color channel
        sums_weights = np.sum(weigths, axis=0)

        # weighted summation of ball center pixels
        means = pixels_center * weigths
        mean = np.sum(means, axis=0)  # is mean for each color

        # normalization for weigths
        mean /= sums_weights

        return mean

    def get_balls_neighborhood(self, image, coordinates_center):
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

        # obtain indices for y and x direction
        coordinates_min = coordinates_center - self.pattern_size * self.num_balls_per_direction
        coordinates_max = coordinates_center + self.pattern_size * self.num_balls_per_direction + 1

        coordinates_balls_y = np.arange(coordinates_min[0], coordinates_max[0], self.pattern_size[0])
        coordinates_balls_x = np.arange(coordinates_min[1], coordinates_max[1], self.pattern_size[1])

        # delete coordinates that are out of bounds
        good_elements_y = np.logical_and(coordinates_balls_y + self.patch_radius < image.shape[0],
                                         coordinates_balls_y - self.patch_radius > 0)
        good_elements_x = np.logical_and(coordinates_balls_x + self.patch_radius < image.shape[1],
                                         coordinates_balls_x - self.patch_radius > 0)

        # # for boundary evaluation
        # max_length_y = num_balls_per_direction * pattern_size[0] + patch_radius
        # max_length_x = num_balls_per_direction * pattern_size[1] + patch_radius

        for index_y in coordinates_balls_y[good_elements_y]:
            for index_x in coordinates_balls_x[good_elements_x]:
                coordinates_ball = np.asarray([index_y, index_x])
                ball = self.get_ball_around_pixel(image, coordinates_ball)

                list_balls.append(ball)
                list_center_pixels.append(image[coordinates_ball[0],
                                                coordinates_ball[1]])

        return np.asarray(list_balls), np.asarray(list_center_pixels)

    def get_camera_parameters(self, path_profile_camera):
        parameters = {}
        with open(path_profile_camera) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                parameters[int(row['color_plane'])] = {'alpha': float(row['alpha']),
                                                       'beta': float(row['beta'])}

        return parameters

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