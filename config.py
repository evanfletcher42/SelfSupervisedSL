"""
This file contains environment-specific paths and parameters which should be set by the user.
"""

# Path to the SceneNet-RGBD training set.
training_set_path = "path/to/training/set/here"

# Path to the SceneNet-RGBD validation set.
validation_set_path = "path/to/validation/set/here"

# Path to a test set, which should be manually created / held out from the above training and validation sets.
# One image from this set will be used to create depth image visualizations during training.
# If None, depth image visualization will be skipped.
visualization_set_path = None
# visualization_set_path = "path/to/test/set/here"
