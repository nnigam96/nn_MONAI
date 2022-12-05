import torch

DATA_DIR = r"/staging/nnigam/inphase"
SIZE = (256, 256) # (Height, Width) of the generated bias field
num_points = 60 # resolution of the simulated coil array

# Uniform Random Variable Bounds
LOW_BOOST_BOUNDS = (0, 0.2)
COIL_VERT_POS_BOUNDS = (SIZE[0] - 1, SIZE[0] * 1.2)
PARAM_B_ADJUST_BOUNDS = (-0.02, 0.02)
COIL_WIDTH_BOUNDS = (0.1, 0.4)
BATCH_SIZE = 1

MRI_DIM = [1, 256, 256, 32]