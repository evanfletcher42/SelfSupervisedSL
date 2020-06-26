from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

class StructuredLightProjectionLayer(Layer):
    """
    Simulates projection of a trainable structured light pattern onto a scene, modeling both camera and scene.

    # This layer outputs:
    # - A projected pattern (just the weights, directly)
    # - A simulated camera image, rendered on top of an input scene.

    """

    # TODO: Docstring ^^
    def __init__(self, image_shape, pattern_shape, pattern_offset, **kwargs):
        # TODO: Docstring
        self.image_shape = image_shape
        self.pattern_shape = pattern_shape
        self.pattern_offset = pattern_offset

        self.pattern = None

        super(StructuredLightProjectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Pattern weights, which will be trained.
        # Pattern is constrained to positive values only (no negative light)
        self.pattern = self.add_weight(name='pattern',
                                       shape=self.pattern_shape,
                                       initializer=tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=0.2),
                                       trainable=True,
                                       constraint=tf.keras.constraints.non_neg()
                                       )

        super(StructuredLightProjectionLayer, self).build(input_shape)

    def get_config(self):
        return{'image_shape': self.image_shape,
               'pattern_shape': self.pattern_shape,
               'pattern_offset': self.pattern_offset}

    def call(self, inputs, **kwargs):
        # Calibration, passed as a tensor to permit input from various cameras & to permit augmentation in SL setup
        calib = inputs[0]  # shape: (sample, n_calib_params)
        fx = calib[..., 0][:, np.newaxis, np.newaxis]  # horizontal focal length
        fy = calib[..., 1][:, np.newaxis, np.newaxis]  # vertical focal length
        cx = calib[..., 2][:, np.newaxis, np.newaxis]  # principal point x
        cy = calib[..., 3][:, np.newaxis, np.newaxis]  # principal point y
        b  = calib[..., 4][:, np.newaxis, np.newaxis]  # structured light baseline in world units
        zr = calib[..., 5][:, np.newaxis, np.newaxis]  # structured light reference plane distance in world units

        depth = inputs[1]  # Depth image of scene.  Shape: (samples, rows, cols)
        # depth_mask = inputs[2]   # Unused in projection.
        ambient = inputs[3]  # Light not from projector.  Shape: (samples, rows, cols)
        albedo = inputs[4]  # Lambertian albedo at each pixel.  Shape: (samples, rows, cols)

        # Create disparity from depth using a simple camera model.
        # Positive disparity => object is closer than reference frame
        disparity = (fx * b / depth) - (fx * b / zr)

        # calibration note: Integer camera coordinates correspond to centers of pixels
        xc = tf.range(self.image_shape[1], dtype=tf.float32)
        yc = tf.range(self.image_shape[0], dtype=tf.float32)

        pixel_grid_x, pixel_grid_y = tf.meshgrid(xc, yc)

        # Unproject to 3D space
        points3d_x = (pixel_grid_x - cx) * depth / fx
        points3d_y = (pixel_grid_y - cy) * depth / fy
        points3d_z = depth
        points3d = tf.stack([points3d_x, points3d_y, points3d_z], axis=-1)

        # Model 1/r^2 falloff with linear distance to points along camera ray.
        dist_falloff = 1.0 / tf.reduce_sum(tf.square(points3d), axis=-1)

        # Prevent exploding gradients; dist_falloff shall not exceed 2.0
        dist_falloff = tf.clip_by_value(dist_falloff, clip_value_min=0.0, clip_value_max=2.0)

        # Compute shadow mask by inspecting disparity.
        #       It is possible to iteratively determine which points are "shaded" / not visible to the projector
        #       by inspecting relative disparities versus pixel distance across rows.
        #       Imagine a foreground-to-background edge which would produce a shadow visible in the camera image.
        #       A given pixel on the background behind the edge is in shadow if the difference in disparity value at
        #       this pixel and at the edge is greater than the distance in pixels between this pixel and the edge.
        #       Intuitively: When the delta disparity == distance between points in pixel space, this pixel and the edge
        #       would lie along a common line with the projector.

        # This search "starts" on the side towards the projector, which is the right side of the image.
        # Since cumulative-max algorithms go the other way, we flip these tensors before and after.

        proj_shadow_test_coords = pixel_grid_x[tf.newaxis, :, :] + tf.reverse(disparity, axis=[2])

        def tf_while_condition(x, loop_counter):
            return tf.not_equal(loop_counter, 0)

        def tf_while_body(x, loop_counter):
            loop_counter -= 1

            # Right-shift x, extending leftmost column
            x_shift = tf.concat((x[..., 0][..., np.newaxis], x[..., :-1]), axis=2)

            # Per element maximum vs not-shifted x
            z = tf.maximum(x, x_shift)

            return z, loop_counter

        # Nothing trainable is behind this cumulative maximum; backprop is disabled here for speed.
        cumulative_max, _ = tf.while_loop(cond=tf_while_condition,
                                          body=tf_while_body,
                                          loop_vars=(proj_shadow_test_coords, self.image_shape[1]),
                                          back_prop=False)

        # Test cumulative max against projector coordinates.
        # Any point where cumulative_max is greater is in shadow; however, if the difference is less than 1 pixel,
        # then the projector pixel is only partially occluded, and the projection should be attenuated accordingly.
        shadow_mask_flip = tf.clip_by_value(proj_shadow_test_coords - cumulative_max + 1.0,
                                            clip_value_min=0.0,
                                            clip_value_max=1.0)

        # Flip shadow mask again to line up with image
        shadow_mask = tf.reverse(shadow_mask_flip, axis=[2])

        # Texture lookup into projected texture (bilinear interpolation)
        projector_coords_x = pixel_grid_x[np.newaxis, :, :] - disparity + self.pattern_offset

        projector_coords_x_floor = tf.math.floor(projector_coords_x)
        projector_coords_x_ceil = tf.math.ceil(projector_coords_x)
        projector_coords_x_frac = projector_coords_x - projector_coords_x_floor

        projector_coords_x_floor = tf.cast(projector_coords_x_floor, dtype=tf.int32)
        projector_coords_x_ceil = tf.cast(projector_coords_x_ceil, dtype=tf.int32)

        # y-coordinates, broadcasted to same number of samples (dim 0) as in disparity
        projector_coords_y = tf.cast(pixel_grid_y[np.newaxis, :, :] + tf.zeros_like(disparity), dtype=tf.int32)

        # create full list of coordinates for gather_nd
        projector_coords_floor = tf.stack([projector_coords_y, projector_coords_x_floor], axis=-1)
        projector_coords_ceil = tf.stack([projector_coords_y, projector_coords_x_ceil], axis=-1)

        # Sample into columns.
        # Note: We want any coordinates here that fall outside the projector texture to return zero intensity.
        # tf.gather_nd will do this, but only on GPU.  On CPU, out-of-bounds indices throw an error.
        # For now, since a zero is the desired output, this should be run on the GPU exclusively.
        projected_pattern_floor = tf.gather_nd(self.pattern, projector_coords_floor)
        projected_pattern_ceil = tf.gather_nd(self.pattern, projector_coords_ceil)

        projected_pattern = projected_pattern_floor * (1.0 - projector_coords_x_frac) + \
            projected_pattern_ceil * projector_coords_x_frac

        # Compose render - attenuations of projected pattern due to shadows, materials, and distance
        projected_pattern_recv = projected_pattern * shadow_mask * albedo * dist_falloff

        # "auto expose" the projector (choose output intensity such that the majority of this scene is exposed)
        projected_pattern_recv *= (0.35 / tf.reduce_mean(projected_pattern_recv, axis=[1, 2], keepdims=True))

        # Add noise.
        read_noise = tf.random.normal(self.image_shape, mean=30.0/1024.0, stddev=10.0/1024.0)

        # composed_render = projected_pattern_recv
        composed_render = ambient + projected_pattern_recv + read_noise

        # Clip composed intensities 0..1 (disabled; stops gradients, even if "realistic.")
        # composed_render = tf.clip_by_value(composed_render, clip_value_min=0.0, clip_value_max=1.0)

        # also pass the pattern directly into the network
        pattern_return = self.pattern.value()[tf.newaxis, ...]

        # Return both the trained reference pattern and the rendered image.
        return [pattern_return, composed_render]
