import numpy as np
from tqdm import tqdm
import logging
import rawpy
from matplotlib import pyplot as plt
from skimage import restoration

class Profiler:
    def __init__(self, patch_radius, num_cores=4):
        self.patch_radius = patch_radius
        self.num_cores = num_cores

    def profile_camera(self, filepath):
        # open image
        image_raw = rawpy.imread(filepath)

        # construct the color planes
        color_planes = self.get_color_planes(image_raw)

        parameters_color_planes = []
        for index_color_plane, color_plane in color_planes.items():
            # obtain the datapoints for patches; std. dev against mean
            datapoints = self.get_std_dev_datapoints(color_plane)

            self.plot_datapoints(datapoints)

            # fit the function
            parameters_color_plane = self.fit_std_dev(datapoints)
            parameters_color_planes.append(parameters_color_plane)

        # todo: irgendwie so zurueckgeben, dass man die parameter der color con rawpy zuordnen kann. am besten per dict.
        return parameters_color_planes

    def plot_datapoints(self, datapoints):
        plt.plot(datapoints['values'], datapoints['standard_deviations'], ".")
        plt.show()


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






