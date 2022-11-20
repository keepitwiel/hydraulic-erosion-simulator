import numpy as np
from numba import njit


@njit
def evaporate(h, r):
    total = np.sum(r)
    dh = total * (h / np.sum(h))
    return h - dh


@njit
def select(arr, dir):
    ny, nx = arr.shape
    y_lb = (1-dir)/2
    y_ub = ny-1+y_lb
    result = arr[y_lb:y_ub, 0:nx]
    return result


@njit
def transport(h, H, dir, k):
    ny, nx = h.shape
    if dir[1] == 0:
        h_source = select(h, dir[0])
        dH = dir[0] * (H[0:ny-1, 0:nx] - H[1:ny, 0:nx])
    elif dir[0] == 0:
        h_source = select(h.T, dir[1]).T
        dH = dir[1] * (H[0:ny, 0:nx-1] - H[0:ny, 1:nx])

    dH = np.maximum(0, dH)
    dh = np.minimum(h_source, k * dH)
    return dh, h_source


@njit
def add_delta(u, du, dir):
    # TODO: simplify by using transpose
    ny, nx = u.shape
    if dir[1] == 0:
        u[0:ny - 1, 0:nx] -= dir[0] * du
        u[1:ny, 0:nx] += dir[0] * du
    elif dir[0] == 0:
        u[0:ny, 0:nx-1] -= dir[1] * du
        u[0:ny, 1:nx] += dir[1] * du
    return u


@njit
def erosion(h_source, dh):
    dz = dh * np.exp(-h_source)
    return dz


# @njit
def update(z, h, k):
    for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        H = z + h
        dh, h_source = transport(h, H, dir, k)
        dz = erosion(h_source, dh)
        h = add_delta(h, dh, dir)
        z = add_delta(z, dz, dir)

    return z, h


class PartialTransportEngine:
    def __init__(self, z, h, k=0.99):
        self.z = z
        self.z0 = z.copy()
        self.h = h
        self.k = k

    def update(self, r):
        self.h += r
        self.z, self.h = update(self.z, self.h, self.k)
        self.h = evaporate(self.h, r)
