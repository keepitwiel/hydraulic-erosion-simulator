import numpy as np
from numba import njit
from matplotlib import pyplot as plt


@njit
def potential_diff(x, dir):
    assert dir in [-1, 1]
    if dir == -1:
        result = x[1:] - x[:-1]
    elif dir == 1:
        result = x[:-1] - x[1:]
    result = np.maximum(0, result)
    return result


@njit
def transport(h, dH, dir):
    assert dir in [-1, 1]
    if dir == -1:
        h_source = h[1:]
    elif dir == 1:
        h_source = h[:-1]

    assert len(h_source) == len(dH)
    dh = np.minimum(h_source, dH / 2)
    if dir == -1:
        h[0:-1] += dh
        h[1:] -= dh
    elif dir == 1:
        h[1:] += dh
        h[:-1] -= dh

    return h, dh


@njit
def update(z, h, dir):
    H = z + h
    dH = potential_diff(H, dir)
    h, dh = transport(h, dH, dir)
    return h, dh, dH

z = np.cumsum(np.random.normal(size=1000))
z -= np.min(z)
z = np.array([np.mean(z[i-50:i+50]) for i in range(50, len(z) - 50)])

h = np.zeros_like(z)
h[len(z) // 4] = 1000.0
h[len(z) // 2] = 1000.0
h[len(z) // 4 + len(z) // 2] = 1000.0


for i in range(100000):
    for dir in [-1, 1]:
        h, dh, dH = update(z, h, dir)
    if i % 10000 == 0:
        plt.plot(z + h)
        plt.plot(z, alpha=0.5)
        # plt.plot(dh * 10 - 10)
        # plt.plot(dH * 10 - 10)
        plt.ylim(min(z) - 5, max(z) + 5)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

