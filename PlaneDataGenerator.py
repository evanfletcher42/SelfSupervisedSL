import numpy as np
import tensorflow.keras as K

'''
Generates planes.  Interface is the same as the SceneNetDataGenerator.   
'''


class PlaneDataGenerator(K.utils.Sequence):
    def __init__(self,  batch_size, shuffle=True):

        self.batch_size = batch_size

        # This generator always makes planes with the below specification.
        self.image_shape = (240, 320)

        self.plane_center_depth = 1.25  # Distance to plane at center of FOV.
        self.plane_angle = 10.0 * np.pi / 180.0

        # plane is uniform in appearance
        self.ambient_intensity = 0.2
        self.plane_albedo = 0.5

    def __len__(self):
        """Used by Keras to determine the number of batches per epoch"""
        return 1

    def __getitem__(self, index):
        """
        Generates one batch of data (reads the appropriate number of images from disk)
        :param index: Batch index.
        :return: (X, y), where X is input and y is labels.
        """

        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        """
        Updates indicies after each epoch.  Used to shuffle data.
        """

        pass

    def __data_generation(self):
        """
        Generates a dataset containing batch_size samples.  These samples are all the same and are always planes.
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

        for i in range(self.batch_size):

            # Camera FOVs for this dataset are specified rather than focal lengths.
            hfov = 60.0 * np.pi / 180.0
            vfov = 45.0 * np.pi / 180.0
            fx = float(self.image_shape[1] / 2.0) / np.tan(hfov / 2.0)
            fy = float(self.image_shape[0] / 2.0) / np.tan(vfov / 2.0)
            cx = float(self.image_shape[1]) / 2.0
            cy = float(self.image_shape[0]) / 2.0

            # Fixed calibration
            sl_baseline = 65.0 / 1000.0  # 65mm baseline
            sl_ref_dist = 0.750  # 75cm reference plane distance

            # Normalized pixel coordinates
            x_norm, y_norm = np.meshgrid((np.arange(self.image_shape[1]) - cx) / fx,
                                         (np.arange(self.image_shape[0]) - cy) / fy
                                        )

            depth = self.plane_center_depth * (1 - x_norm * np.tan(self.plane_angle))

            # depth is always valid
            depth_mask = np.ones_like(depth)

            calib = [fx, fy, cx, cy, sl_baseline, sl_ref_dist]

            # Generate ground-truth disparity given this calibration
            disparity = (fx * sl_baseline / depth) - (fx * sl_baseline / sl_ref_dist)

            ambient = np.ones_like(depth) * self.ambient_intensity
            albedo = np.ones_like(depth) * self.plane_albedo

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
