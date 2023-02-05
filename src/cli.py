import numpy as np
from tqdm import tqdm

from gui_utils import initialize_engine


SEED = 116
MAP_SIZE = 1024
SEA_LEVEL = 74
RANDOM_AMPLITUDE = 32
SLOPE_AMPLITUDE = 256
SMOOTHNESS = 0.6
RAINFALL = 0.01
N_ITERS = 2000
DT = 0.1
K_C = 0.00001
K_E = 0.0001
EROSION_FLAG = False


if __name__ == "__main__":
    engine = initialize_engine(
        SEED, MAP_SIZE, SEA_LEVEL, RANDOM_AMPLITUDE, SLOPE_AMPLITUDE, SMOOTHNESS, RAINFALL,
    )
    for _ in tqdm(range(N_ITERS)):
        engine.update(DT, K_C, K_E, EROSION_FLAG)

    filename = f"/tmp/world_{MAP_SIZE}.npz"
    print(filename)
    np.savez(filename, land=engine.z, water=engine.h, rainfall=engine.r)
