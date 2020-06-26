import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Activation, Add
from group_norm import GroupNormalization
import robust_loss.adaptive


# ============= Network Blocks =============
def resnet_block(n_filters, input, dilation_rate=(1, 1)):
    """
    Pre-activation ResNet block with group normalization.
    :param n_filters: Number of filters ("depth") of this convolution
    :param input: Keras functional object representing input to this resnet block.
    :return: Keras functional object representing output of this resnet block.
    """

    n_in = GroupNormalization(groups=n_filters // 4)(input)
    a_in = Activation(tf.nn.leaky_relu)(n_in)

    c1 = Conv2D(filters=n_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                use_bias=False)(a_in)
    n1 = GroupNormalization(groups=n_filters // 4)(c1)
    a1 = Activation(tf.nn.leaky_relu)(n1)

    c2 = Conv2D(filters=n_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                use_bias=False)(a1)

    s = Add()([input, c2])

    return s


def separable_resnet_block(n_filters, input, dilation_rate=(1, 1)):
    """
    Puts together a ResNet block with two _separable_ convolutions.
    :param n_filters: Number of filters ("depth") of this convolution
    :param input: Keras functional object representing input to this resnet block.
    :param dilation_rate: Tuple of 2 integers, specifying the dilation rate for dilated convolution.
    :return: Tensor, representing the output of this block.
    """
    n_in = GroupNormalization(groups=n_filters // 4)(input)
    a_in = Activation(tf.nn.leaky_relu)(n_in)

    c1 = SeparableConv2D(filters=n_filters,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=False)(a_in)
    n1 = GroupNormalization(groups=n_filters // 4)(c1)
    a1 = Activation(tf.nn.leaky_relu)(n1)

    c2 = SeparableConv2D(filters=n_filters,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=False)(a1)

    s = Add()([input, c2])

    return s


# =========== Loss Functions ===============

# Adaptive loss.
# 3 separate losses for 3 outputs, each maintaining their own parameters.  Otherwise, these are the same.
adaptive_loss_1 = robust_loss.adaptive.AdaptiveLossFunction(num_channels=240*320,
                                                            float_dtype=tf.float32)
adaptive_loss_2 = robust_loss.adaptive.AdaptiveLossFunction(num_channels=240*320,
                                                            float_dtype=tf.float32)
adaptive_loss_3 = robust_loss.adaptive.AdaptiveLossFunction(num_channels=240*320,
                                                            float_dtype=tf.float32)


def mask_aware_robust_1(yTrue, yPred):
    """
    Keras loss function wrapper for mask-aware adaptive robust loss, which zeros out residuals in masked-out areas
    before passing data to the loss function.
    :param yTrue: Ground truth depth
    :param yPred: Prediction (including passed-through mask data)
    :return: Result of adaptive loss.
    """

    # For non-image-like loss functions
    yTrue_d = tf.reshape(yTrue, (-1, 240 * 320))  # Required; TF needs to know the last channel size is 1.
    yPred_d = tf.reshape(yPred[..., 0], (-1, 240 * 320))
    yPred_mask = tf.reshape(yPred[..., 1], (-1, 240 * 320))

    resid = yPred_d - yTrue_d
    # Masked regions should not contribute to the loss or gradients.
    # Easiest way to do this is to make the residual zero at these locations.
    resid *= yPred_mask

    return adaptive_loss_1(resid)


def mask_aware_robust_2(yTrue, yPred):
    """
    Keras loss function wrapper for mask-aware adaptive robust loss, which zeros out residuals in masked-out areas
    before passing data to the loss function.
    :param yTrue: Ground truth depth
    :param yPred: Prediction (including passed-through mask data)
    :return: Result of adaptive loss.
    """

    # For non-image-like loss functions
    yTrue_d = tf.reshape(yTrue, (-1, 240 * 320))  # Required; TF needs to know the last channel size is 1.
    yPred_d = tf.reshape(yPred[..., 0], (-1, 240 * 320))
    yPred_mask = tf.reshape(yPred[..., 1], (-1, 240 * 320))

    resid = yPred_d - yTrue_d
    # Masked regions should not contribute to the loss or gradients.
    # Easiest way to do this is to make the residual zero at these locations.
    resid *= yPred_mask

    return adaptive_loss_2(resid)


def mask_aware_robust_3(yTrue, yPred):
    """
    Keras loss function wrapper for mask-aware adaptive robust loss, which zeros out residuals in masked-out areas
    before passing data to the loss function.
    :param yTrue: Ground truth depth
    :param yPred: Prediction (including passed-through mask data)
    :return: Result of adaptive loss.
    """

    # For non-image-like loss functions
    yTrue_d = tf.reshape(yTrue, (-1, 240 * 320))  # Required; TF needs to know the last channel size is 1.
    yPred_d = tf.reshape(yPred[..., 0], (-1, 240 * 320))
    yPred_mask = tf.reshape(yPred[..., 1], (-1, 240 * 320))

    resid = yPred_d - yTrue_d
    # Masked regions should not contribute to the loss or gradients.
    # Easiest way to do this is to make the residual zero at these locations.
    resid *= yPred_mask

    return adaptive_loss_3(resid)
