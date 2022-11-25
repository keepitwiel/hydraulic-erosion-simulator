import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from tqdm import tqdm

@njit
def diamond_square(p: int, smoothness: float, random_seed: int) -> np.ndarray:
    np.random.seed(random_seed)
    n = 2**p + 1
    z = np.zeros((n, n))

    for q in range(p, 0, -1):
        d = 2**q  # step size
        h = 2 ** (q - 1)  # half step

        # random corners
        r = np.random.normal(0, d**smoothness, (1 + 2 ** (p - q), 1 + 2 ** (p - q)))

        # add to grid
        z[0::d, 0::d] += r

        # interpolate edges
        z[h:n:d, 0:n:d] = 0.5 * z[0 : n - d : d, 0:n:d] + 0.5 * z[d:n:d, 0:n:d]
        z[0:n:d, h:n:d] = 0.5 * z[0:n:d, 0 : n - d : d] + 0.5 * z[0:n:d, d:n:d]

        # interpolate middle
        z[h:n:d, h:n:d] = 0.25 * (
            z[0 : n - d : d, 0 : n - d : d]
            + z[0 : n - d : d, d:n:d]
            + z[d:n:d, 0 : n - d : d]
            + z[d:n:d, d:n:d]
        )

    return z


def generate_height_map(height: int, width: int, random_seed: int, smoothness=0.6):
    max_dimension = max(height, width)
    p = np.ceil(np.log2(max_dimension)).astype(int)
    z = diamond_square(p, smoothness, random_seed)
    z_max = np.max(z)
    z = z[:height, :width] / z_max * 128
    return z


@njit
def update(z, h):
    dh = np.zeros_like(h)
    H = z + h
    for i in range(1, h.shape[1] - 1):
        for j in range(1, h.shape[1] - 1):
            if h[i, j] > 0:
                delta_down = max(H[i, j] - H[i, j - 1], 0) / 4
                delta_up = max(H[i, j] - H[i, j + 1], 0) / 4
                delta_left = max(H[i, j] - H[i - 1, j], 0) / 4
                delta_right = max(H[i, j] - H[i + 1, j], 0) / 4

                dh[i, j - 1] += delta_down
                dh[i, j + 1] += delta_up
                dh[i - 1, j] += delta_left
                dh[i + 1, j] += delta_right

                # substract height difference from middle
                dh[i, j] -= (delta_down + delta_up + delta_left + delta_right)

    return dh


def main():
    z = generate_height_map(128, 128, 42)
    h = np.zeros_like(z)  # water height above ground
    h[5:25, 5:25] = 500 - z[5:25, 5:25]

    vmax = np.max(z)

    def generator():
        while True:
            yield

    i = 0
    for _ in tqdm(generator()):
        dh = update(z, h)
        h += 1 * dh

        if i % 100 == 0:
            print(i, np.sum(h))
            plt.imshow(z + h, vmin=np.min(z), vmax=vmax)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

        i += 1


if __name__ == "__main__":
    main()
