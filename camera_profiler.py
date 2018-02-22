import numpy as np
import rawpy
from matplotlib import pyplot as plt
from skimage import restoration
from scipy.optimize import curve_fit
from common_functions import ascombe_transform_scale
import os
from csv import DictWriter

class Profiler:
    def __init__(self, patch_radius, num_cores=4):
        self.patch_radius = patch_radius

    def profile_camera(self, filepath_input, output_dir_plots):
        # open image
        image_raw = rawpy.imread(filepath_input)

        # construct the color planes
        color_planes_pseudo = self.get_color_planes_pseudo_indices(image_raw)

        # obtain the datapoints (std.dev vs. mean) for all pseudo-indiced color planes
        datapoints_pseudo = {}
        for index_color_plane, color_plane in color_planes_pseudo.items():
            # obtain the datapoints for patches; std. dev against mean
            datapoints_color_plane = self.get_std_dev_datapoints(color_plane)
            datapoints_pseudo[index_color_plane] = datapoints_color_plane

        # merge all datapoints of pseudo-color planes that belong to the same sensor color
        mapping_color_indices = self.get_mapping_raw_pattern(image_raw)
        datapoints_merged = {}
        for index_pseudo_color, index_sensor_color in mapping_color_indices.items():
            if index_sensor_color not in datapoints_merged:
                datapoints_merged[index_sensor_color] = datapoints_pseudo[index_pseudo_color]
            else:
                datapoints_merged[index_sensor_color]['values'].extend(datapoints_pseudo[index_pseudo_color]['values'])
                datapoints_merged[index_sensor_color]['standard_deviations'].extend(
                    datapoints_pseudo[index_pseudo_color]['standard_deviations'])

        # perform the fitting for the datapoints
        parameters_color_planes = {}
        for index_color_plane, datapoints_color_plane in datapoints_merged.items():
            # fit the function
            parameters_color_plane = self.fit_std_dev(datapoints_color_plane)
            parameters_color_planes[index_color_plane] = parameters_color_plane

            self.plot_datapoints(datapoints_color_plane,
                                 parameters_color_plane,
                                 index_color_plane,
                                 output_dir_plots,
                                 "{}.png")

#       # plot results of variance stabilization
        for index_color_plane, color_plane in color_planes_pseudo.items():
            # index is index for pseudo color plane, but parameters are for real colors.
            # find parameters of real color for the color plane
            index_sensor_color = mapping_color_indices[index_color_plane]
            parameters_color_plane = parameters_color_planes[index_sensor_color]

            # transform and plot result
            plane_transformed = ascombe_transform_scale(color_plane,
                                                        parameters_color_plane['alpha'],
                                                        parameters_color_plane['beta'])
            # remove nans
            plane_transformed[np.isnan(plane_transformed)] = 0

            datapoints = self.get_std_dev_datapoints(plane_transformed)
            self.plot_datapoints(datapoints, parameters_color_plane, index_color_plane,
                                 output_dir_plots, "{}_transformed.png",  plot_function=False)

        return parameters_color_planes

    def plot_datapoints(self, datapoints, parameters_color_plane, index_color_plane, output_dir_plots, name_pattern,
                        plot_function=True):
        plt.plot(datapoints['values'], datapoints['standard_deviations'], ".")

        if plot_function:
            values_fit = np.linspace(0, np.amax(datapoints['values']), 500)
            fit = self.function_conversion_sigma(values_fit,
                                                 parameters_color_plane['alpha'],
                                                 parameters_color_plane['beta'])
            plt.plot(values_fit, fit, '-')

        output_path = os.path.join(output_dir_plots, name_pattern.format(index_color_plane))
        plt.savefig(output_path)
        plt.close()


    def get_color_planes_pseudo_indices(self, image_raw):
        """ obtain array with size of image, where each value corresponds to the index of the color matrix color;
            only visible area to prevent garbage data at borders

        :param image_raw: rawpy raw image
        :return: dict with rawpy color indices as keys and color planes as numpy arrays as values
        """


        image = image_raw.raw_image_visible
        raw_pattern_pseudo_indices = self.get_raw_pattern_unique(image_raw)

        # trim the image so that pattern repeats completely in all axes
        shape_planes = np.floor((np.asarray(image.shape) / np.asarray(raw_pattern_pseudo_indices.shape))).astype(np.int)
        shape_image_trimmed = shape_planes * np.asarray(raw_pattern_pseudo_indices.shape)
        image_trimmed = image[0:shape_image_trimmed[0], 0:shape_image_trimmed[1]]

        # build pseudo-color-indices image
        padding = shape_image_trimmed - np.asarray(raw_pattern_pseudo_indices.shape)
        color_indices_pseudo = np.pad(raw_pattern_pseudo_indices,[(0, padding[0]), (0, padding[1])], 'wrap')

        color_planes = {}
        for color_index in raw_pattern_pseudo_indices.flatten():
            pixels_color = image_trimmed[color_indices_pseudo == color_index]
            color_planes[color_index] = pixels_color.reshape(shape_planes)

        return color_planes

    def get_raw_pattern_unique(self, image_raw):
        """ Obtain raw pattern with upcounting pseudo color-indices

        :param image_raw:
        :return:
        """
        raw_pattern = image_raw.raw_pattern
        indices = np.arange(0, raw_pattern.size)

        return np.reshape(indices, raw_pattern.shape)

    def get_mapping_raw_pattern(self, image_raw):
        """ Obtain mapping from pseudo-indices to indices

        :param image_raw:
        :return:
        """

        raw_pattern = image_raw.raw_pattern
        raw_pattern_pseudo = self.get_raw_pattern_unique(image_raw)

        mapping = {}
        for index_raw, index_pseudo in zip(raw_pattern.flatten(), raw_pattern_pseudo.flatten()):
            mapping[index_pseudo] = index_raw

        return mapping

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
                # and if there are no invalid pixels
                if np.amax(patch) < max_value and np.amin(patch) > min_value and not np.isnan(patch).any():
                    standard_deviation = restoration.estimate_sigma(patch)
                    mean = np.mean(patch)

                    standard_deviations.append(standard_deviation)
                    values.append(mean)

        return {'standard_deviations': standard_deviations,
                'values': values}

    def get_patch_around_pixel(self, image, index_y, index_x):
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

    def save_parameters(self, parameters, output_file):
        with open(output_file, 'w') as csvfile:
            fieldnames = ['color_plane', 'alpha', 'beta']
            writer = DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for index_color_plane, parameters_color_plane in parameters.items():
                row_dict = {'color_plane': index_color_plane,
                            'alpha': parameters_color_plane['alpha'],
                            'beta': parameters_color_plane['beta']}
                writer.writerow(row_dict)
