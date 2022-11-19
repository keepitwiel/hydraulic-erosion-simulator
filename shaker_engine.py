import numpy as np
from numba import njit


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

    return dh


@njit
def add_delta(x, dx, dir):
    if dir == (-1, 0):
        x[:-1, :] += dx
        x[1:, :] -= dx
    elif dir == (1, 0):
        x[1:, :] += dx
        x[:-1, :] -= dx
    elif dir == (0, -1):
        x[:, :-1] += dx
        x[:, 1:] -= dx
    elif dir == (0, 1):
        x[:, 1:] += dx
        x[:, :-1] -= dx
    return x


@njit
def erosion(h, dh, dir):
    if dir == (-1, 0):
        hh = h[1:, :]
    elif dir == (1, 0):
        hh = h[:-1, :]
    elif dir == (0, -1):
        hh = h[:, 1:]
    elif dir == (0, 1):
        hh = h[:, :-1]
    dz = 0.1 * dh / np.maximum(0.1, hh)
    return dz


@njit
def update(z, h, k):
    for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        H = z + h
        dh = transport(h, H, dir, k)
        dz = erosion(h, dh, dir)
        h = add_delta(h, dh, dir)
        z = add_delta(z, dz, dir)

    return z, h


class ShakerEngine:
    def __init__(self, z, h, k=0.99):
        self.z = z
        self.z0 = z.copy()
        self.h = h
        self.k = k

    def update(self, r):
        self.h += r
        self.z, self.h = update(self.z, self.h, self.k)
