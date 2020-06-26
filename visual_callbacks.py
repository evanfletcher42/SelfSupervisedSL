import cv2
import tensorflow as tf
import tensorflow.keras.callbacks
import numpy as np
import os


class DisparityPlotter(tf.keras.callbacks.Callback):
    """
    Predicts and outputs disparity as calculated by the model.  
    """

    def __init__(self, generator, save_path):
        self.generator = generator
        self.save_path = save_path
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        super(DisparityPlotter, self).__init__()

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        # For the sake of plot consistency, always run the same image.
        x_data, y_true = self.generator[0]

        yPred_refined, yPred_unrefined, yPred_gm_net = self.model.predict(x_data)

        # always display the first image in the batch
        yTrue_d = y_true[0][0, :, :]

        yPred_refined_d = yPred_refined[0][:, :, 0]
        yPred_unrefined_d = yPred_unrefined[0][:, :, 0]
        yPred_gm_net_d = yPred_gm_net[0][:, :, 0]

        mask = yPred_refined[0][:, :, 1]

        min_d = np.min(yTrue_d)
        max_d = np.max(yTrue_d)

        # Unrefined image ("direct")
        yPred_unrefined_d = np.clip((yPred_unrefined_d - min_d) / (max_d - min_d) * 255.0, 0.0, 255.0)
        yPred_unrefined_rgb = cv2.applyColorMap(yPred_unrefined_d.astype(np.uint8), cv2.COLORMAP_PARULA)
        yPred_unrefined_rgb *= mask.astype(np.uint8)[..., np.newaxis]

        # After cost volume refinement
        yPred_gm_net_d = np.clip((yPred_gm_net_d - min_d) / (max_d - min_d) * 255.0, 0.0, 255.0)
        yPred_gm_net_rgb = cv2.applyColorMap(yPred_gm_net_d.astype(np.uint8), cv2.COLORMAP_PARULA)
        yPred_gm_net_rgb *= mask.astype(np.uint8)[..., np.newaxis]

        # refined image:
        yPred_refined_d = np.clip((yPred_refined_d - min_d) / (max_d - min_d) * 255.0, 0.0, 255.0)
        yPred_refined_rgb = cv2.applyColorMap(yPred_refined_d.astype(np.uint8), cv2.COLORMAP_PARULA)
        yPred_refined_rgb *= mask.astype(np.uint8)[..., np.newaxis]

        # Ground truth
        yTrue_d = np.clip((yTrue_d - min_d) / (max_d - min_d) * 255.0, 0.0, 255.0)
        yTrue_rgb = cv2.applyColorMap(yTrue_d.astype(np.uint8), cv2.COLORMAP_PARULA)
        yTrue_rgb *= mask.astype(np.uint8)[..., np.newaxis]

        # Compose for display
        display_img = np.zeros((yTrue_d.shape[0] * 4, yTrue_d.shape[1], 3))

        display_img[0*yTrue_d.shape[0]:1*yTrue_d.shape[0], :, :] = yPred_unrefined_rgb
        display_img[1*yTrue_d.shape[0]:2*yTrue_d.shape[0], :, :] = yPred_gm_net_rgb
        display_img[2*yTrue_d.shape[0]:3*yTrue_d.shape[0], :, :] = yPred_refined_rgb
        display_img[3*yTrue_d.shape[0]:4*yTrue_d.shape[0], :, :] = yTrue_rgb

        cv2.imwrite(os.path.join(self.save_path, "disparity_%05d.png" % epoch), display_img)


class DrawPatternCallback(tf.keras.callbacks.Callback):
    """
    Periodically saves out the learned pattern in the SLProjection layer as an image.
    """
    def __init__(self, save_path):
        self.save_path = save_path
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        self.batches_seen = 0
        super(DrawPatternCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # self.batches_seen += 1

        # if self.batches_seen % 100 == 1:
        # Write out the weights of the pattern layer as an image.
        pattern_weights = np.array(self.model.get_layer("SLProjection").get_weights())[0]
        pattern_weights = pattern_weights / np.max(pattern_weights)
        pattern_weights = np.clip(pattern_weights, 0.0, 1.0) * 255.0
        pattern_weights = pattern_weights.astype(np.uint8)
        cv2.imwrite(os.path.join("pattern", "pattern_%05d.png" % epoch), pattern_weights)