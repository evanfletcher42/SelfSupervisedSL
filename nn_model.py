import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Softmax, UpSampling2D, Concatenate, Lambda
from group_norm import GroupNormalization
from CostVolumeLayer import CostVolumeLayer
from StructuredLightProjectionLayer import StructuredLightProjectionLayer
from tensorflow.keras.models import Model
import numpy as np
from utils import resnet_block, separable_resnet_block


class nn_model:
    def __init__(self):
        # Hyperparameters

        # input shape (rows, cols, channels)
        self.image_shape = (240, 320)

        # Disparity search range and resolution.
        self.disparities = np.arange(start=-24,
                                     stop=36,
                                     step=1)

        print("Explicitly matching disparities:")
        print(self.disparities)

        # Reference image is horizontally wider than the aux image, such that every pixel in the aux image can always
        # map to a pixel in the reference image, regardless of disparity.
        # (Note that this requires more complicated calibration)
        self.ref_image_shape = (self.image_shape[0],
                                self.image_shape[1] + (np.max(self.disparities) - np.min(self.disparities)))

        # Principal point offset between camera and projector, since projector image is larger
        self.ref_h_offset = np.max(self.disparities)

        # Siamese tower
        self.tower_n_channels = 32

        # Various constructed models go here
        self.tower_model = None
        self.disparity_refinement_model = None

        # overall model goes here
        self.model = None

    def setup_siamese_towers(self):
        """
        Builds tf.keras.models.Model object that acts as a tower feature extractor.
        Modeled after ActiveStereoNet.
        :return: None
        """

        # Input is either the ref or aux images.  These both have the same number of rows and channels, but the ref
        # image has more columns.
        img_in = Input(shape=(self.image_shape[0], None, 1))

        conv_01 = Conv2D(filters=self.tower_n_channels,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(img_in)

        last_input = conv_01
        for i in range(7):
            resnet = resnet_block(self.tower_n_channels, last_input)
            last_input = resnet

        # One final convolution w/o activation
        conv_final = Conv2D(filters=self.tower_n_channels,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(last_input)

        # collect model
        self.tower_model = Model(img_in, conv_final)

    def setup_disparity_refinement_net(self):

        # Input image and high-res disparity map are the same size
        img_in = Input(shape=(self.image_shape[0], self.image_shape[1], 32))
        disp_in = Input(shape=(self.image_shape[0], self.image_shape[1], 1))

        # Image branch
        img_c1 = Conv2D(filters=16,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same')(img_in)
        img_bn1 = GroupNormalization(groups=4)(img_c1)
        img_a1 = Activation(tf.nn.leaky_relu)(img_bn1)
        img_rn1 = resnet_block(n_filters=16, input=img_a1)
        img_rn2 = resnet_block(n_filters=16, dilation_rate=(2, 2), input=img_rn1)

        # Disparity branch
        disp_c1 = Conv2D(filters=16,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(disp_in)
        disp_bn1 = GroupNormalization(groups=4)(disp_c1)
        disp_a1 = Activation(tf.nn.leaky_relu)(disp_bn1)
        disp_rn1 = resnet_block(n_filters=16, input=disp_a1)
        disp_rn2 = resnet_block(n_filters=16, dilation_rate=(2, 2), input=disp_rn1)

        # Glue these together along feature axis
        ct1 = Concatenate(axis=-1)([img_rn2, disp_rn2])

        # Some more processing
        ref_rn1 = resnet_block(n_filters=32, dilation_rate=(4, 4), input=ct1)
        ref_rn2 = resnet_block(n_filters=32, dilation_rate=(8, 8), input=ref_rn1)
        ref_rn3 = resnet_block(n_filters=32, input=ref_rn2)
        ref_rn4 = resnet_block(n_filters=32, input=ref_rn3)

        # Reduce to single channel output (disparity residual)
        disp_resid_out = Conv2D(filters=1,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same')(ref_rn4)

        self.disparity_refinement_model = Model([img_in, disp_in], disp_resid_out)

    def setup_model(self):

        calib_input = Input(shape=(6,))
        depth_input = Input(shape=self.image_shape)
        depth_mask_input = Input(shape=self.image_shape)
        ambient_input = Input(shape=self.image_shape)
        albedo_input = Input(shape=self.image_shape)

        # Render synthetic structured light image
        ref_img, aux_img = StructuredLightProjectionLayer(pattern_shape=self.ref_image_shape,
                                                          image_shape=self.image_shape,
                                                          pattern_offset=self.ref_h_offset,
                                                          name="SLProjection")([calib_input,
                                                                                depth_input,
                                                                                depth_mask_input,
                                                                                ambient_input,
                                                                                albedo_input])

        # add channel axis to images for later convolutions
        ref_img = ref_img[..., tf.newaxis]
        aux_img = aux_img[..., tf.newaxis]

        # Set up various network branches
        # TODO: In a practical application, the reference image should be (somewhat) fixed; consider pre-computing
        #       descriptors for the reference image.
        # TODO: Modalities of projected pattern and imaged scene are not the same.  Why use a siamese network?
        self.setup_siamese_towers()
        self.setup_disparity_refinement_net()

        ref_descriptors = self.tower_model(ref_img)
        aux_descriptors = self.tower_model(aux_img)

        # Create cost volume
        cost_vol_raw = CostVolumeLayer(aux_image_shape=(self.image_shape[0], self.image_shape[1]),
                                       disparities=self.disparities
                                       )([ref_descriptors, aux_descriptors])

        # Condition cost volume (softmax)
        cost_vol = Softmax(axis=-1)(cost_vol_raw)

        # Create an output here for the un-refined cost volume.  Helps train the siamese towers easier.
        r_full = tf.constant(self.disparities, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
        disp_unrefined = tf.reduce_sum(cost_vol * r_full, axis=-1, keepdims=True)
        concat_unrefined_out = Concatenate(axis=-1, name="unrefined_output")([disp_unrefined, depth_mask_input[..., tf.newaxis]])

        # Perform some separable convolutions on the cost volume.
        # Mixes information spatially, to a degree.  Sort of like learning a semi-global matching function.
        # Separable convolutions, since there's (probably) a lot of disparities/channels in the cost volume.
        cost_vol_c = cost_vol
        for i in range(7):
            cost_vol_c = separable_resnet_block(len(self.disparities), cost_vol_c)
        cost_vol_a = Softmax(axis=-1)(cost_vol_c)

        # Add an output here to better train this cost volume refinement net
        disp_gm_net = tf.reduce_sum(cost_vol_a * r_full, axis=-1, keepdims=True)
        concat_gm_net_out = Concatenate(axis=-1, name="gm_net_output")([disp_gm_net, depth_mask_input[..., tf.newaxis]])

        # Normalize disparities to 0..1 range in soft-argmax to keep things well-conditioned.
        r = tf.constant((self.disparities - np.min(self.disparities)) / (np.max(self.disparities) - np.min(self.disparities)),
                        dtype=tf.float32
                        )[tf.newaxis, tf.newaxis, tf.newaxis, :]

        disp_initial = tf.reduce_sum(cost_vol_a * r, axis=-1, keepdims=True)

        # ASN-like disparity refinement network, which produces a residual to be applied to the upsampled disparity
        disp_residual = self.disparity_refinement_model([aux_descriptors, disp_initial])

        # Add residual into upsampled disparity image to produce final disparity
        disp_img_out = Add()([disp_initial, disp_residual])

        # Scale up output disparities appropriately
        disp_scaled_out = Lambda(lambda x: x * (np.max(self.disparities) - np.min(self.disparities)) + np.min(self.disparities),
                                 name="disparity_scaled")(disp_img_out)

        # Concatenate with input depth mask for loss calculation
        concat_img_out = Concatenate(axis=-1, name="refined_output")([disp_scaled_out, depth_mask_input[..., tf.newaxis]],)

        # Build Model mapping [ref, aux] inputs to disparity
        self.model = Model([calib_input,
                            depth_input,
                            depth_mask_input,
                            ambient_input,
                            albedo_input],

                           [concat_img_out, concat_unrefined_out, concat_gm_net_out])

        print(self.model.summary())

        return self.model
