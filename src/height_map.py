import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter


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


def  generate_height_map(height: int, width: int, random_seed: int, smoothness=0.6, final_blur=2.0):
    max_dimension = max(height, width)
    p = np.ceil(np.log2(max_dimension)).astype(int)
    z = diamond_square(p, smoothness, random_seed)
    z = z[:height, :width]
    z = gaussian_filter(z, sigma=final_blur)

    # normalize
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    return z
