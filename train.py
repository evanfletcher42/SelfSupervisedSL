import tensorflow as tf
import numpy as np
import nn_model
from SceneNetDataGenerator import SceneNetDataGenerator
from visual_callbacks import DisparityPlotter, DrawPatternCallback
from utils import mask_aware_robust_1, mask_aware_robust_2, mask_aware_robust_3
import os
import config

# Repeatability
np.random.seed(1337)
tf.random.set_seed(1337)

# Setup
batch_size = 1
epochs = 10000000  # Will never practically finish.  Instead, user should inspect losses for convergence.

# Callbacks
dp_cb = DrawPatternCallback(save_path="pattern")

tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_grads=False,
                                       write_images=True,
                                       update_freq='epoch',
                                       profile_batch=0)

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

ckpt_cb = tf.keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}.hdf5',
                                             verbose=0,
                                             save_best_only=False,
                                             save_weights_only=True,
                                             mode='auto',
                                             period=5)

callback_list = [tb_cb, ckpt_cb, dp_cb]

# dataset generation
training_generator = SceneNetDataGenerator(data_path=config.training_set_path,
                                           batch_size=batch_size)
validation_generator = SceneNetDataGenerator(data_path=config.validation_set_path,
                                             batch_size=batch_size)

if config.visualization_set_path is not None:
    minimal_generator = SceneNetDataGenerator(data_path=config.validation_set_path,
                                              batch_size=1)
    disp_plt_cb = DisparityPlotter(generator=minimal_generator,
                                   save_path="disparity_plots")
    callback_list.append(disp_plt_cb)

# Model setup
nn_m = nn_model.nn_model()
model = nn_m.setup_model()

model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0, clipvalue=0.5),
              loss={"unrefined_output": mask_aware_robust_1,
                    "refined_output": mask_aware_robust_2,
                    "gm_net_output": mask_aware_robust_3}
              )

# Fit the model.
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    steps_per_epoch=500,  # dataset is massive (5 million images); don't go though all every epoch
                    validation_steps=100,
                    use_multiprocessing=False,
                    callbacks=callback_list
                    )
