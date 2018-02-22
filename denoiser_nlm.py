import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import logging
from csv import DictReader
from copy import deepcopy
from common_functions import ascombe_transform_scale, inverse_ascombe_transform_scale

class Denoiser:
    def __init__(self, patch_radius, h, num_balls_per_direction, pattern_size=None, path_profile_camera=None,
                 num_cores=4, rotate_patches=False):
        """

        :param patch_radius:
        :param h:
        :param num_balls_per_direction: Number of patches to search around each pixel in each direction (l, r, u, d)
            number of patches generated will be (2 * num_balls_per_direction)**2
        :param pattern_size:
        :param path_profile_camera:
        :param num_cores:
        """
        # store some parameters in class just so that they can be attached easily in multiprocessing.
        self.patch_radius = patch_radius
        self.h = h
        self.num_balls_per_direction = num_balls_per_direction
        self.pattern_size = pattern_size
        self.num_cores = num_cores
        self.rotate_patches = rotate_patches

        if path_profile_camera is not None:
            self.parameters_camera = self.get_camera_parameters(path_profile_camera)
        else:
            self.parameters_camera = None

    def filter_image(self, image_raw, slice_denoise=None):
        """ Filter the image with pattern-aware NL means.

        :param image_raw: a rawpy raw image
        :param slice_denoise: an np slice object. Optional to define region of image to be filtered
            for faster testing
        :return:
        """
        # store old pattern size
        pattern_size_old = deepcopy(self.pattern_size)

        # set new pattern size
        self.pattern_size = np.asarray(image_raw.raw_pattern.shape)

        # copy the image
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

        # write filtered data into image
        if slice_denoise is not None:
            image_data_transformed[slice_denoise] = image_data_filtered
        else:
            image_data_transformed = image_data_filtered

        # re-transform image data
        image_data_filtered_backtransformed = self.ascombe_transform_data(image_data_transformed, image_raw, inverse=True)

        # write filtered data into image
        image_raw.raw_image[...] = image_data_filtered_backtransformed.astype(np.uint16)

        # reset patter size
        self.pattern_size = pattern_size_old

        return image_raw

    def ascombe_transform_data(self, image_data, image_raw, inverse=False):
        """  Perform ascombe transform in order to stabilize variance

        :param image_data:
        :param image_raw:
        :param inverse:
        :return:
        """
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
        """ Filter the image with non-local means

        :param image: image data as numpy array
        :return: filtered image
        """
        # get all patches to evaluate
        logging.info("Calculating filtered image")
        # balls, pixels_center = get_all_balls_image(image, patch_radius)

        # obtain all possible reelative shifts of locations of a pixel's neighbor patches
        num_balls = 2* self.num_balls_per_direction + 1
        min_shift_y = - self.pattern_size[0] * self.num_balls_per_direction
        min_shift_x = - self.pattern_size[1] * self.num_balls_per_direction
        shifts_y = np.arange(min_shift_y,
                             -min_shift_y + self.pattern_size[0],
                             step=self.pattern_size[0],
                             dtype=np.int)
        shifts_x = np.arange(min_shift_x,
                             -min_shift_x + self.pattern_size[1],
                             step=self.pattern_size[1],
                             dtype=np.int)

        image_filtered = np.zeros_like(image)
        sums_weights = np.zeros_like(image)

        pool = Pool(processes=self.num_cores)
        for shift_y in tqdm(shifts_y):
            if self.num_cores > 1:
                data_multiprocessing = []
                for shift_x in shifts_x:
                    data_multiprocessing.append({'image': image,
                                                 'shift_y': shift_y,
                                                 'shift_x': shift_x})


                results_multiprocessing = pool.map(self.get_weights_column, data_multiprocessing)

                for image_shifted, weights in results_multiprocessing:
                    sums_weights += weights
                    image_filtered += weights * image_shifted
            else:
                for shift_x in shifts_x:
                    image_shifted = np.roll(image, [shift_y, shift_x], (0,1))
                    weights = self.calculate_weights(image, image_shifted)
                    sums_weights += weights
                    image_filtered += weights * image_shifted

        # normalize weights
        image_filtered /= sums_weights

        return image_filtered

    def get_weights_column(self, data):
        shift_x = data['shift_x']
        shift_y = data['shift_y']
        image = data['image']
        image_shifted = np.roll(image, [shift_y, shift_x], (0, 1))
        weights = self.calculate_weights(image, image_shifted)

        return image_shifted, weights


    def calculate_weights(self, image, image_shifted):
        square_differences = np.power(image - image_shifted, 2)
        distances = self.calculate_distances(square_differences)
        exponent = - np.divide(distances, self.h**2)

        return np.exp(exponent)

    def calculate_distances(self, square_differences):
        """ Calculare distances as l2-norm, normalized by patch pixel count

        :param square_differences:
        :return:
        """
        # square patches, so use same shifts for x and y
        shifts = np.arange(- self.patch_radius, self.patch_radius+1, step=1, dtype=np.int)

        distances = np.zeros_like(square_differences)
        for shift_y in shifts:
            shifted_y = np.roll(square_differences, shift_y, 0)
            for shift_x in shifts:
                if shift_x != 0:
                    distances[shift_x:] += shifted_y[:-shift_x]
                    distances[:shift_x] += shifted_y[-shift_x:]
                else:
                    distances += shifted_y

        # normalize by division by patch pixel count
        distances /= (2*self.patch_radius + 1)**2

        return distances

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

    def get_camera_parameters(self, path_profile_camera):
        """ Load photometric parameters from camera profile

        :param path_profile_camera:
        :return:
        """
        parameters = {}
        with open(path_profile_camera) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                parameters[int(row['color_plane'])] = {'alpha': float(row['alpha']),
                                                       'beta': float(row['beta'])}

        return parameters
