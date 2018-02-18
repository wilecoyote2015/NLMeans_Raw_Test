import numpy as np
from tqdm import tqdm
import logging
import rawpy
from matplotlib import pyplot as plt
from skimage import restoration
from scipy.optimize import curve_fit
import os, sys

class Profiler:
    def __init__(self, patch_radius, num_cores=4):
        self.patch_radius = patch_radius
        self.num_cores = num_cores

    def profile_camera(self, filepath_input, output_dir_plots):
        # open image
        image_raw = rawpy.imread(filepath_input)

        # construct the color planes
        color_planes = self.get_color_planes(image_raw)

        parameters_color_planes = {}
        for index_color_plane, color_plane in color_planes.items():
            # obtain the datapoints for patches; std. dev against mean
            datapoints = self.get_std_dev_datapoints(color_plane)

            # fit the function
            parameters_color_plane = self.fit_std_dev(datapoints)
            parameters_color_planes[index_color_plane] = parameters_color_plane

            self.plot_datapoints(datapoints, parameters_color_plane, index_color_plane, output_dir_plots)

        return parameters_color_planes

    def plot_datapoints(self, datapoints, parameters_color_plane, index_color_plane, output_dir_plots,
                        plot_function=True):
        plt.plot(datapoints['values'], datapoints['standard_deviations'], ".")

        if plot_function:
            values_fit = np.linspace(0, np.amax(datapoints['values']), 500)
            fit = self.function_conversion_sigma(values_fit,
                                                 parameters_color_plane['alpha'],
                                                 parameters_color_plane['beta'])
            plt.plot(values_fit, fit, '-')

        output_path = os.path.join(output_dir_plots, "{}.png".format(index_color_plane))
        plt.savefig(output_path)
        plt.close()


    def get_color_planes(self, image_raw):
        # obtain array with size of image, where each value corresponds to the index of the color matrix color;
        # only visible area to prevent garbage data at borders

        image = image_raw.raw_image_visible
        raw_pattern = image_raw.raw_pattern

        # trim the image so that pattern repeats completely in all axes
        shape_planes = np.floor((np.asarray(image.shape) / np.asarray(raw_pattern.shape))).astype(np.int)
        shape_image_trimmed = shape_planes * np.asarray(raw_pattern.shape)
        image_trimmed = image[0:shape_image_trimmed[0], 0:shape_image_trimmed[1]]
        color_indices_trimmed = image_raw.raw_colors_visible[0:shape_image_trimmed[0], 0:shape_image_trimmed[1]]

        color_planes = {}
        for color_index in raw_pattern.flatten():
            pixels_color = image_trimmed[color_indices_trimmed == color_index]
            color_planes[color_index] = pixels_color.reshape(shape_planes)

        return color_planes

    def get_std_dev_datapoints(self, color_plane):

        max_value = np.amax(color_plane)
        min_value = np.amin(color_plane)

        # iterate over all patches and obtain standard deviation and mean value
        standard_deviations = []
        values = []
        for index_y in range(self.patch_radius,
                             color_plane.shape[0] - self.patch_radius,
                             self.patch_radius * 2 + 1):
            for index_x in range(self.patch_radius,
                                 color_plane.shape[1] - self.patch_radius,
                                 self.patch_radius * 2 + 1):
                patch = self.get_patch_around_pixel(color_plane, index_y, index_x)

                # only proceed if patch does not contain the max value of the plane, which would indicate clipping
                if np.amax(patch) < max_value and np.amin(patch) > min_value:  # todo: min or 0?
                    # todo: use more sophisticated algorithm from scipy
                    standard_deviation = restoration.estimate_sigma(patch)
                    mean = np.mean(patch)

                    standard_deviations.append(standard_deviation)
                    values.append(mean)

        return {'standard_deviations': standard_deviations,
                'values': values}

    def get_patch_around_pixel(self, image, index_y, index_x):
        # coordinates_min = coordinates - patch_radius
        # coordinates_max = coordinates + patch_radius + 1  # +1 because of np slicing

        return image[index_y - self.patch_radius:index_y + self.patch_radius + 1,
                     index_x - self.patch_radius:index_x + self.patch_radius + 1]

    def fit_std_dev(self, datapoints):
        parameters, covariances = curve_fit(self.function_conversion_sigma,
                                            datapoints['values'],
                                            datapoints['standard_deviations'])
        return {'alpha': parameters[0],
                'beta': parameters[1]}

    def function_conversion_sigma(self, mean_measured, scale_alpha, shift_beta):
        sigma_measured = np.sqrt(np.maximum(scale_alpha * (mean_measured - shift_beta), 0))  # np.max to prevent negative

        return sigma_measured



