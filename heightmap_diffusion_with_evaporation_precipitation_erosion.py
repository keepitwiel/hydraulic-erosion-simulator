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


def generate_height_map(height: int, width: int, random_seed: int):
    max_dimension = max(height, width)
    smoothness = 0.6
    p = np.ceil(np.log2(max_dimension)).astype(int)
    z = diamond_square(p, smoothness, random_seed)
    z_max = np.max(z)
    z = z[:height, :width] / z_max * 128
    return z


def evaporation(z, h):
    return 0.001 * h


def precipitation(z, evap):
    """
    Calculate precipitation of all evaporated water
    :param z:
    :param evap:
    :return:
    """
    total = np.sum(evap)
    mask = z > 50
    result = total * mask / np.sum(mask)
    return result


@njit
def diffusion(z, h, active):
    dh = np.zeros_like(h)
    flux = np.zeros_like(h)

    H = z + h
    for i in range(1, h.shape[1] - 1):
        for j in range(1, h.shape[1] - 1):
                if h[i, j] > 0 and active[i, j] == True:
                    delta_down = max(H[i, j] - H[i, j - 1], 0) / 4
                    delta_up = max(H[i, j] - H[i, j + 1], 0) / 4
                    delta_left = max(H[i, j] - H[i - 1, j], 0) / 4
                    delta_right = max(H[i, j] - H[i + 1, j], 0) / 4

                    # water outflow from tile (i, j) to neighbors
                    dh[i, j - 1] += delta_down / 2
                    dh[i, j + 1] += delta_up / 2
                    dh[i - 1, j] += delta_left / 2
                    dh[i + 1, j] += delta_right / 2

                    # flux: total mass flowing out
                    flux[i, j] = (delta_down + delta_up + delta_left + delta_right) / 2

                    # substract flux from water mass at tile (i, j)
                    # to account for conservation of mass
                    dh[i, j] -= flux[i, j]

    return dh, flux


def main():
    fig, axes = plt.subplots(1, 4)

    # z = np.array([[max(x, y) for x in range(128)] for y in range(128)], dtype=float)
    z = generate_height_map(512, 512, 42)
    active = np.ones_like(z, dtype=bool)
    erosion = np.zeros_like(z)
    h = -np.minimum(0.0, z)  # water height above ground
    # precip = np.zeros_like(z)
    # precip[80:, 100:] = 0.0001

    def generator():
        while True:
            yield

    i = 0
    for _ in tqdm(generator()):
        precip = 0.0001 * (z > 10)  # np.random.exponential(0.0001, size=z.shape)
        # evap = evaporation(z, h)
        # h -= evap  # substract evaporated water from surface water
        # h += precipitation(z, 10)  # add back evaporated water in form of precipitation
        h += precip
        dh, flux = diffusion(z - erosion, h, active)  # diffuse surface water
        active = np.abs(dh) > 0
        gx, gy = np.gradient(z - erosion)
        slope = np.sqrt(gx**2 + gy**2)
        erosion += 0.1 * flux / (1.0 + h**2) * slope  # erosion
        h += dh
        h[h > 10] -= np.sum(precip) * h[h > 10] / np.sum(h[h > 10])

        if i % 100 == 0:
            axes[0].set_title(f"iteration {i}: active: {np.sum(active)}, flux: {np.sum(flux):6.0f}, water mass {np.sum(h):6.0f}")
            axes[0].imshow(z - erosion + h, vmin=0, vmax=128)
            axes[1].imshow(slope, vmin=-10, vmax=10)
            axes[2].imshow(h, vmin=0, vmax=10)
            axes[3].imshow(erosion, vmin=0, vmax=10)
            plt.draw()
            plt.pause(0.0001)

        i += 1


if __name__ == "__main__":
    main()
