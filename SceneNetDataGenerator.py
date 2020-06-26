import numpy as np
import tensorflow.keras as K
import os
import cv2
from perlin2d import generate_perlin_noise_2d

'''
Generates batch data for Keras from the SceneNet RGB-D dataset.   
'''


class SceneNetDataGenerator(K.utils.Sequence):
    def __init__(self,  data_path, batch_size, shuffle=True):

        self.batch_size = batch_size

        # Find all PNG images at the specified path
        print("Enumerating images in dataset path " + data_path + "...")
        rgb_images_list = []

        for root, subfolders, files in os.walk(data_path):
            if os.path.split(root)[-1] == "photo":
                print(root)
                for file in files:
                    rgb_images_list.append(os.path.join(root, file))

        print("Found %d images" % len(rgb_images_list))

        # Create list of tuples of paths to data files: (rgb, depth, depth_mask, normals)
        self.data_paths = []

        print("Processing paths...")
        for rgb_img_path in rgb_images_list:

            depth_path = rgb_img_path.replace("photo", "depth")
            depth_path = depth_path.replace("jpg", "png")

            self.data_paths.append((rgb_img_path, depth_path))

        print("Done")
        self.indices = np.arange(len(self.data_paths))
        self.shuffle = shuffle

        pass

    def __len__(self):
        """Used by Keras to determine the number of batches per epoch"""
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of data (reads the appropriate number of images from disk)
        :param index: Batch index.
        :return: (X, y), where X is input and y is labels.
        """

        indices_this_batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        path_sets_this_batch = [self.data_paths[k] for k in indices_this_batch]

        X, y = self.__data_generation(path_sets_this_batch)

        return X, y

    def on_epoch_end(self):
        """
        Updates indicies after each epoch.  Used to shuffle data.
        """

        if self.shuffle:
            np.random.shuffle(self.indices)

        pass

    def __data_generation(self, path_sets):
        """
        Generates a dataset containing batch_size samples.
        As this will be used with a synthetic renderer, structured-light baseline and reference plane distance are
        randomized, as a data-augmentation measure.
        :param path_sets: List of tuples of paths to data files to be loaded from disk.
        :return: (X, y), where X is input and y is labels for this batch
        """

        # inputs
        calibrations = []
        ambient_images = []
        depth_images = []
        depth_masks = []
        albedo_images = []

        # outputs
        disparity_images = []

        # Generate a Perlin noise LUT which will be used to modulate reflectivity of objects by their color
        perlin_img = generate_perlin_noise_2d(shape=(256, 256), res=(4, 4))

        for (rgb_image_path, depth_path) in path_sets:

            # RGB images are direct from PNG, and are normalized 0..1
            rgb_image_raw = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
            rgb_image = rgb_image_raw.astype(np.float) / 255.0

            # Camera FOVs for this dataset are specified rather than focal lengths.
            hfov = 60.0 * np.pi / 180.0
            vfov = 45.0 * np.pi / 180.0
            fx = float(rgb_image.shape[1] / 2.0) / np.tan(hfov / 2.0)
            fy = float(rgb_image.shape[0] / 2.0) / np.tan(vfov / 2.0)
            cx = float(rgb_image.shape[1]) / 2.0
            cy = float(rgb_image.shape[0]) / 2.0

            # Fixed calibration
            sl_baseline = 65.0 / 1000.0  # 65mm baseline
            sl_ref_dist = 0.750  # 75cm reference plane distance

            # Depth images are 16 bit integer _ray length_ in millimeters; convert to meters
            depth_ray_length = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0

            # Convert to z-plane depth
            u, v = np.meshgrid(np.arange(rgb_image.shape[1]) - cx,
                               np.arange(rgb_image.shape[0]) - cy)

            xp = u * 1.0 / fx
            yp = v * 1.0 / fy
            zp = np.ones_like(xp)
            xyzp = np.concatenate([xp[..., np.newaxis], yp[..., np.newaxis], zp[..., np.newaxis]], axis=-1)

            xyz_ray = xyzp / np.linalg.norm(xyzp, axis=-1, keepdims=True)

            sample_xyz = xyz_ray * depth_ray_length[..., np.newaxis]

            depth = sample_xyz[..., -1]

            # You'd think in a synthetic dataset, every point would have valid depth, but here we are.
            # I'm assuming they used Blender for this and some rays struck the background, returning zero depth.
            depth_mask = np.ones_like(depth)
            min_depth = fx * sl_baseline / ((fx * sl_baseline / sl_ref_dist) + 35.0)
            depth_mask[depth_ray_length < min_depth] = 0.0

            # Modify depth mask: Any points that are masked out should be "far away" to avoid casting
            # erroneous shadows across the scene.
            depth[depth_mask == 0] = 10000.0

            calib = [fx, fy, cx, cy, sl_baseline, sl_ref_dist]

            # Generate ground-truth disparity given this calibration
            disparity = (fx * sl_baseline / depth) - (fx * sl_baseline / sl_ref_dist)

            # Additionally mask out points where it would not be possible for this sensor configuration to be correct
            depth_mask[disparity < -24] = 0.0
            depth_mask[disparity > 35] = 0.0

            # RGB -> Ambient intensity (here, a direct grayscale conversion)
            ambient = 0.2125 * rgb_image[..., 0] + \
                      0.7154 * rgb_image[..., 1] + \
                      0.0721 * rgb_image[..., 2]

            # Attenuate ambient light by a random-ish amount
            ambient = ambient * (0.05 + np.random.rand() * 0.1)

            # Create albedo maps for the render
            # RGB -> YCrCb, use CrCb planes to sample into perlin noise.
            # When rendered, ought to result in visible changes to returned intensity that seem correlated with objects.
            ycbcr_img = cv2.cvtColor(rgb_image_raw, cv2.COLOR_RGB2YCrCb)

            # World reflectance at pattern wavelength is assumed to be 18% +/- 5%
            albedo = 0.18 + 0.1 * (perlin_img[ycbcr_img[:, :, 1], ycbcr_img[:, :, 2]] - 0.5)

            # Collect data for this sample
            calibrations.append(calib)
            ambient_images.append(ambient)
            depth_images.append(depth)
            depth_masks.append(depth_mask)
            albedo_images.append(albedo)

            disparity_images.append(disparity)

        calibrations = np.array(calibrations)
        ambient_images = np.array(ambient_images)
        depth_images = np.array(depth_images)
        depth_masks = np.array(depth_masks)
        albedo_images = np.array(albedo_images)
        disparity_images = np.array(disparity_images)

        return [calibrations, depth_images, depth_masks, ambient_images, albedo_images], \
               [disparity_images, disparity_images, disparity_images]
