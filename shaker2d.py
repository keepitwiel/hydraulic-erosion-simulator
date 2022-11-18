import numpy as np
from numba import njit
from matplotlib import pyplot as plt

from heightmap_diffusion import generate_height_map


@njit
def transport(h, H, dir, k):
    # assert dir in DIRECTIONS
    # TODO: simplify
    if dir == (-1, 0):
        dH = H[1:, :] - H[:-1, :]
        h_source = h[1:, :]
    elif dir == (1, 0):
        dH = H[:-1, :] - H[1:, :]
        h_source = h[:-1, :]
    elif dir == (0, -1):
        dH = H[:, 1:] - H[:, :-1]
        h_source = h[:, 1:]
    elif dir == (0, 1):
        dH = H[:, :-1] - H[:, 1:]
        h_source = h[:, :-1]

    dH = np.maximum(0, dH)
    # assert h_source.shape == dH.shape

    dh = np.minimum(h_source, k * dH)
    if dir == (-1, 0):
        h[:-1, :] += dh
        h[1:, :] -= dh
    elif dir == (1, 0):
        h[1:, :] += dh
        h[:-1, :] -= dh
    elif dir == (0, -1):
        h[:, :-1] += dh
        h[:, 1:] -= dh
    elif dir == (0, 1):
        h[:, 1:] += dh
        h[:, :-1] -= dh

    return h


@njit
def update(z, h, k):
    for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        # assert len(z.shape) == 2
        H = z + h
        h = transport(h, H, dir, k)

    return z, h


if __name__ == "__main__":
    # for seed in range(10):
    # print(f"seed: {seed}")
    np.random.seed(42)
    z = generate_height_map(128, 128, 42)
    h = np.zeros_like(z)
    h[::8, ::8] = 400.0

    k = 0.99

    for i in range(100000):
        h = update(z, h, k)
        if i % 10 == 0:
            plt.imshow(z + h, vmin=0, vmax=100)
            # plt.bar(x=range(len(z[::10])), bottom=z[::10], height=h[::10], width=1)
            # plt.bar(x=range(len(z[::10])), bottom=0, height=z[::10], alpha=0.5, width=1)
            # plt.ylim(min(z) - 5, max(z) + 5)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
