import tensorflow as tf
import numpy as np
import nn_model
from SceneNetDataGenerator import SceneNetDataGenerator
from PlaneDataGenerator import PlaneDataGenerator
import matplotlib.pyplot as plt
from utils import mask_aware_robust_1, mask_aware_robust_2, mask_aware_robust_3

"""
Utility script, which can perform various evaluations on a trained network.
This script will:
    - Export reconstructed depth maps (at each output) as an OBJ point cloud
    - Visualize, and evaluate the RMS error of, depth images at each output
    - Perform evaluation on a dataset scene, or optionally on a slanted uniform plane
    
Settings at the beginning of the file should be modified by the user as needed.  
"""

# ========== User Settings ===========

# Path to model weights to load / evaluate.
model_weights_path = "checkpoints/weights.830.hdf5"

# If True, evaluate using a plane.
# If False, evaluate on a scene from a selected dataset.
eval_with_plane = False

# Dataset path, if evaluating on a dataset sample.
eval_dataset_path = "path/to/eval/set/here"

# ========= End User Settings ==========

# Repeatability
np.random.seed(1337)
tf.random.set_seed(1337)


def disparity_img_to_file(disparity, vmin, vmax, cmap, file_path):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    img = cmap(norm(disparity))

    plt.imsave(file_path, img)


def disparity_to_obj(disparity, f, B, zr, obj_file_path):
    # disparity -> depth
    z = f * B / ((f * B) / zr + disparity)

    # reproject assuming cx, cy @ center of image
    xp, yp = np.meshgrid(np.arange(disparity.shape[1]) - disparity.shape[1] / 2,
                         np.arange(disparity.shape[0]) - disparity.shape[0] / 2)

    p_xyz = np.zeros((disparity.shape[0], disparity.shape[1], 3))
    p_xyz[:, :, 0] = xp / f * z
    p_xyz[:, :, 1] = yp / f * z
    p_xyz[:, :, 2] = z

    p_xyz = p_xyz.reshape((p_xyz.shape[0] * p_xyz.shape[1], p_xyz.shape[2]))

    with open(obj_file_path, 'w') as f:
        f.write("# OBJ file\n")
        for v in p_xyz:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))

        print("Exported OBJ file: " + obj_file_path)


# Setup
batch_size = 1

# dataset generation
if eval_with_plane:
    eval_generator = PlaneDataGenerator(batch_size=1)
else:
    eval_generator = SceneNetDataGenerator(data_path=eval_dataset_path,
                                           batch_size=1)

nn_m = nn_model.nn_model()
model = nn_m.setup_model()

model.load_weights(model_weights_path)

model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0, clipvalue=0.5),
              loss={"unrefined_output": mask_aware_robust_1,
                    "refined_output": mask_aware_robust_2,
                    "gm_net_output": mask_aware_robust_3},
              )

# Pull an image from the generator
x_data, y_true = eval_generator[0]

# Forward prediction using the trained network
yPred_refined, yPred_unrefined, yPred_gm_net = model.predict_generator(generator=eval_generator)

# always display the first image in the batch
yTrue_d = y_true[0][0, :, :]

yPred_refined_d = yPred_refined[0][:, :, 0]
yPred_unrefined_d = yPred_unrefined[0][:, :, 0]
yPred_gm_net_d = yPred_gm_net[0][:, :, 0]

mask = yPred_refined[0][:, :, 1]

# Compute numeric results
print("Results: ")
print("Unrefined Disparity RMSE (px): " + str(np.sqrt(np.mean(np.square(yPred_unrefined_d - yTrue_d)))))
print("GM Net Disparity RMSE (px):    " + str(np.sqrt(np.mean(np.square(yPred_gm_net_d - yTrue_d)))))
print("Refined Disparity RMSE (px):   " + str(np.sqrt(np.mean(np.square(yPred_refined_d - yTrue_d)))))

# Write point clouds as .OBJ files for visualization
# Camera FOVs for this dataset are specified rather than focal lengths.
hfov = 60.0 * np.pi / 180.0
vfov = 45.0 * np.pi / 180.0
fx = float(yTrue_d.shape[1] / 2.0) / np.tan(hfov / 2.0)
fy = float(yTrue_d.shape[0] / 2.0) / np.tan(vfov / 2.0)
# Fixed structured-light system calibration
sl_baseline = 65.0 / 1000.0  # 65mm baseline
sl_ref_dist = 0.750  # 75cm reference plane distance

disparity_to_obj(yPred_unrefined_d, fx, sl_baseline, sl_ref_dist, "01_unrefined.obj")
disparity_to_obj(yPred_gm_net_d, fx, sl_baseline, sl_ref_dist, "02_gm_net.obj")
disparity_to_obj(yPred_refined_d, fx, sl_baseline, sl_ref_dist, "03_refined.obj")
disparity_to_obj(yTrue_d, fx, sl_baseline, sl_ref_dist, "99_true.obj")

# Plot things

# Configure plot range: Start with the range of possible disparities, compress if results contain something smaller
plot_vmin = np.min(nn_m.disparities)
plot_vmax = np.max(nn_m.disparities)

plot_vmin = np.max([plot_vmin,
                    np.min(yPred_unrefined_d),
                    np.min(yPred_gm_net_d),
                    np.min(yPred_refined_d),
                    np.min(yTrue_d)
                    ])

plot_vmax = np.min([plot_vmax,
                    np.max(yPred_unrefined_d),
                    np.max(yPred_gm_net_d),
                    np.max(yPred_refined_d),
                    np.max(yTrue_d)
                    ])

# write to files
disparity_img_to_file(yPred_unrefined_d, plot_vmin, plot_vmax, plt.cm.viridis, "00_raw_disparity.png")
disparity_img_to_file(yPred_gm_net_d, plot_vmin, plot_vmax, plt.cm.viridis, "01_gm_refined_disparity.png")
disparity_img_to_file(yPred_refined_d, plot_vmin, plot_vmax, plt.cm.viridis, "02_final_disparity.png")
disparity_img_to_file(yTrue_d, plot_vmin, plot_vmax, plt.cm.viridis, "99_true_disparity.png")

# make visible plots

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# pull colorbar info from unrefined plot, which is probably noisiest
im = ax[0, 0].imshow(yPred_unrefined_d, vmin=plot_vmin, vmax=plot_vmax)
ax[0, 0].set_title("Unrefined Predicted Disparity")

ax[0, 1].imshow(yPred_gm_net_d, vmin=plot_vmin, vmax=plot_vmax)
ax[0, 1].set_title("GM Net Predicted Disparity")

ax[1, 0].imshow(yPred_refined_d, vmin=plot_vmin, vmax=plot_vmax)
ax[1, 0].set_title("Refined Predicted Disparity")

ax[1, 1].imshow(yTrue_d, vmin=plot_vmin, vmax=plot_vmax)
ax[1, 1].set_title("Ground Truth Disparity")

# Single colorbar for all plots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()

# Plot residuals
resid_vmin = -2.0
resid_vmax = 2.0

# write to files
disparity_img_to_file(yPred_unrefined_d - yTrue_d, resid_vmin, resid_vmax, plt.cm.seismic, "00_raw_disparity_resid.png")
disparity_img_to_file(yPred_gm_net_d - yTrue_d, resid_vmin, resid_vmax,  plt.cm.seismic, "01_gm_refined_disparity_resid.png")
disparity_img_to_file(yPred_refined_d - yTrue_d, resid_vmin, resid_vmax,  plt.cm.seismic, "02_final_disparity_resid.png")

# make visible plots

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# pull colorbar info from unrefined plot, which is probably noisiest
im = ax[0, 0].imshow(yPred_unrefined_d - yTrue_d, vmin=resid_vmin, vmax=resid_vmax, cmap='seismic')
ax[0, 0].set_title("Unrefined Predicted Disparity Error")

ax[0, 1].imshow(yPred_gm_net_d - yTrue_d, vmin=resid_vmin, vmax=resid_vmax, cmap='seismic')
ax[0, 1].set_title("GM Net Predicted Disparity Error")

ax[1, 0].imshow(yPred_refined_d - yTrue_d, vmin=resid_vmin, vmax=resid_vmax, cmap='seismic')
ax[1, 0].set_title("Refined Predicted Disparity Error")

ax[1, 1].imshow(yTrue_d - yTrue_d, vmin=resid_vmin, vmax=resid_vmax, cmap='seismic')
ax[1, 1].set_title("Ground Truth Disparity Error")

# Single colorbar for all plots
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()
