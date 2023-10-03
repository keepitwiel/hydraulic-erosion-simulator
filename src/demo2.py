import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from numba import njit

from height_map import generate_height_map


SEED = 116
MAP_WIDTH = 64
MAP_HEIGHT = 128
SEA_LEVEL = 64
RANDOM_AMPLITUDE = 16
SLOPE_AMPLITUDE = 256

SMOOTHNESS = 0.6
RAINFALL = 1.0
N_ITERS = 1000
DT = 0.1
K_C = 0.1
K_S = 0.1
K_D = 0.1
K_E = 0.0001


@njit
def update(z, h, dh, r, dt=0.1):
    dh += r
    height, width = z.shape
    for k in range(100):
        for j in range(1, height-1):
            for i in range(1, width-1):
                h_ = h[j, i]
                z_ = z[j, i]

                if h_ > 0:
                    left = z[j, i-1] + h[j, i-1] < z_ + h_ if i - 1 > 0 else False
                    right = z[j, i+1] + h[j, i+1] < z_ + h_ if i + 1 < width else False
                    up = z[j-1, i] + h[j-1, i] < z_ + h_ if j - 1 > 0 else False
                    down = z[j+1, i] + h[j+1, i] < z_ + h_ if j + 1 < width else False

                    total = left + right + up + down

                    if total > 0:
                        delta = h_ / total * dt
                        dh[j, i] -= h_ * dt
                        if left:
                            dh[j, i-1] += delta
                        if right:
                            dh[j, i+1] += delta
                        if up:
                            dh[j-1, i] += delta
                        if down:
                            dh[j+1, i] += delta

        h += dh
        dh.fill(0.0)


def main():
    z = generate_height_map(MAP_HEIGHT, MAP_WIDTH, SEED, SMOOTHNESS) * RANDOM_AMPLITUDE
    z += np.linspace(0, 1, MAP_HEIGHT).reshape(-1, 1).dot(np.ones(MAP_WIDTH).reshape(1, -1)) * SLOPE_AMPLITUDE
    z -= SEA_LEVEL
    r = np.zeros_like(z)
    r[MAP_HEIGHT - 8, MAP_WIDTH // 2 - 8] = RAINFALL
    #h = -np.minimum(0.0, z)
    h = np.zeros_like(z)
    dh = np.zeros_like(h)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(z)

    for _ in tqdm(range(1000)):
        update(z, h, dh, r)
        axes[1].imshow(h, vmax=RAINFALL * 0.1)
        axes[2].imshow(h)
        axes[1].set_title(f"{np.sum(h):2.0f}")
        plt.pause(0.001)


if __name__ == "__main__":
    main()
