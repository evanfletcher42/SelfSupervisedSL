from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np

"""
Performs a cost-volume aggregation operation on input images of arbitrary channel depth.  
"""

class CostVolumeLayer(Layer):
    def __init__(self, aux_image_shape, disparities, **kwargs):
        super(CostVolumeLayer, self).__init__(**kwargs)

        # Shape of the aux image over which we will be matching the reference image
        self.aux_image_shape = aux_image_shape

        # Disparities to search
        self.disparities = disparities

        # Offset to apply to the reference image (permits negative disparities).
        self.ref_offset = np.max(disparities)

    def build(self, input_shape):
        # This layer serves only to perform a matching operation and contains no trainable weights.  So, nothing here.
        super(CostVolumeLayer, self).build(input_shape)

    def get_config(self):
        return {"aux_image_shape": self.aux_image_shape,
                "disparities": self.disparities}

    def call(self, inputs, **kwargs):
        # Perform the matching operation.
        # input: list of two 4d tensors with shape: (samples, rows, cols, channels)
        # output: 4D tensor with shape (samples, rows, cols, 1)

        ref = inputs[0]
        aux = inputs[1]

        # Cost volume output
        cost_volume_list = []

        # Perform matching operation in a loop, which will _probably_ get parallelized when the compute graph is built
        # worst case, if memory is a problem, force this to be done iteratively in a tf while loop
        for d in self.disparities:
            # Slice a view of the reference image at the specified disparity.
            # Positive disparities mean the object is closer to the camera, and will cause the pattern to appear to
            # shift right; this is equivalent to sampling a smaller column index.

            ref_start = self.ref_offset - d
            ref_end = ref_start + self.aux_image_shape[1]

            # should be the same size as aux
            shifted_ref = ref[..., ref_start:ref_end, :]

            # Matching algorithm = dot product between tensors.  Positive/big for well correlated signals.
            # Shape: (samples, rows, cols)
            cost = tf.reduce_sum(shifted_ref * aux, axis=-1)

            # stick on the end of the cost volume
            cost_volume_list.append(cost)

        # assemble cost volume - shape: (samples, rows, cols, len(self.disparities))
        cost_volume = tf.stack(cost_volume_list, axis=-1)

        return cost_volume

    def compute_output_shape(self, input_shape):
        out_shape = (input_shape[1][0], input_shape[1][1], input_shape[1][2], len(self.disparities))
        return out_shape
