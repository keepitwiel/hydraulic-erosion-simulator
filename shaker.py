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
def transport(h, dH, dir, k):
    assert dir in [-1, 1]
    if dir == -1:
        h_source = h[1:]
    elif dir == 1:
        h_source = h[:-1]

    assert len(h_source) == len(dH)
    dh = np.minimum(h_source, k * dH)
    if dir == -1:
        h[0:-1] += dh
        h[1:] -= dh
    elif dir == 1:
        h[1:] += dh
        h[:-1] -= dh

    return h, dh


@njit
def update(z, h, dir, k):
    H = z + h
    dH = potential_diff(H, dir)
    h, dh = transport(h, dH, dir, k)
    return h, dh, dH


k = 0.99

for seed in range(10):
    print(f"seed: {seed}")
    np.random.seed(seed)
    z = np.cumsum(np.random.normal(size=1000))
    z -= np.min(z)
    z = np.array([np.mean(z[i-50:i+50]) for i in range(50, len(z) - 50)])

    h = np.zeros_like(z)
    h[::10] = 10
    # h[len(z) // 4] = 1000.0
    # h[len(z) // 2] = 1000.0
    # h[len(z) // 4 + len(z) // 2] = 1000.0


    for i in range(10000):
        for dir in [-1, 1]:
            h, dh, dH = update(z, h, dir, k)
        if i % 1000 == 0:
            plt.bar(x=range(len(z[::10])), bottom=z[::10], height=h[::10], width=1)
            plt.bar(x=range(len(z[::10])), bottom=0, height=z[::10], alpha=0.5, width=1)
            plt.ylim(min(z) - 5, max(z) + 5)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

