import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from numba import njit

from height_map import generate_height_map


SEED = 116
MAP_WIDTH = 64
MAP_HEIGHT = 128
SEA_LEVEL = 64
RANDOM_AMPLITUDE = 64
SLOPE_AMPLITUDE = 256
SMOOTHNESS = 0.6
RAINFALL = 0.1
EVAP = 0.001
EROSION = 0.1


def update(z, dz, h, dh, flux, flux_out, r, evap=EVAP, erosion=EROSION, dt=0.1):
    update_water(z, h, dh, flux, r, dt)
    diffuse_terrain(z, dz, h, flux, erosion, dt)
    evaporate(h, flux, evap, dt)
    flux_out[:, :] = flux[:, :]
    flux.fill(0.0)


@njit
def update_water(z, h, dh, flux, r, dt):
    dh += r
    height, width = z.shape
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
                    dh[j, i-1] += left * delta
                    dh[j, i+1] += right * delta
                    dh[j-1, i] += up * delta
                    dh[j+1, i] += down * delta

                    flux[j, i] += h_ * dt
                    flux[j, i-1] += left * delta
                    flux[j, i+1] += right * delta
                    flux[j-1, i] += up * delta
                    flux[j+1, i] += down * delta

    h += dh
    dh.fill(0.0)


@njit
def diffuse_terrain(z, dz, h, flux, erosion, dt):
    height, width = z.shape
    for j in range(1, height-1):
        for i in range(1, width-1):
            delta = flux[j, i] / (h[j, i]**2 + 1) * erosion * dt
            dz[j, i] -= delta * 4
            dz[j, i - 1] += delta
            dz[j, i + 1] += delta
            dz[j - 1, i] += delta
            dz[j + 1, i] += delta
    z += dz
    dz.fill(0.0)


@njit
def evaporate(h, flux, evap, dt):
    delta = np.minimum(evap * dt, h / (flux + 1))
    h -= np.minimum(h, delta)


def main():
    z = generate_height_map(MAP_HEIGHT, MAP_WIDTH, SEED, SMOOTHNESS) * RANDOM_AMPLITUDE
    z += np.linspace(0, 1, MAP_HEIGHT).reshape(-1, 1).dot(np.ones(MAP_WIDTH).reshape(1, -1)) * SLOPE_AMPLITUDE
    z -= SEA_LEVEL
    dz = np.zeros_like(z)
    z0 = z.copy()

    r = np.zeros_like(z)
    # r[1:MAP_HEIGHT-1, 1:MAP_WIDTH-1] = RAINFALL / (MAP_HEIGHT * MAP_WIDTH)
    r[MAP_HEIGHT - 8, MAP_WIDTH // 2 - 8] = RAINFALL
    h = -np.minimum(0.0, z)
    dh = np.zeros_like(h)
    flux = np.zeros_like(h)
    flux_out = np.zeros_like(h)

    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(z)
    axes[0].set_title("Terrain altitude")
    axes[1].set_title(f"Total water: {np.sum(h):2.2f}")
    axes[2].set_title("Flux")
    axes[3].set_title("Terrain displacement")

    for _ in tqdm(range(10000)):
        update(z, dz, h, dh, flux, flux_out, r, evap=EVAP, erosion=EROSION, dt=0.1)

        if _ % 100 == 0:
            axes[1].imshow(h, vmax=RAINFALL)
            axes[1].set_title(f"Total water: {np.sum(h):2.2f}")
            axes[2].imshow(flux_out, vmax=RAINFALL)
            axes[3].imshow(z - z0, vmin=-1, vmax=1)
            plt.pause(0.001)


if __name__ == "__main__":
    main()
