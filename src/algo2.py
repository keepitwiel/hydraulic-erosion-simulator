from numba import njit
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


@njit
def get_active(mh, mw, z, H, i, j):
    s = 0
    if j == 0:
        active_left = False
    elif H[i, j - 1] < z[i, j]:
        active_left = False
    else:
        active_left = True
        s += 1

    if j == mw - 1:
        active_right = False
    elif H[i, j + 1] < z[i, j]:
        active_right = False
    else:
        active_right = True
        s += 1

    if i == 0:
        active_up = False
    elif H[i-1, j] < z[i, j]:
        active_up = False
    else:
        active_up = True
        s += 1

    if i == mh - 1:
        active_down = False
    elif H[i+1, j] < z[i, j]:
        active_down = False
    else:
        active_down = True
        s += 1

    return s, active_left, active_right, active_up, active_down


@njit
def get_height_diff(active, H, h, i, j, di, dj):
    dh = 0.0
    if active:
        i_ = i + di
        j_ = j + dj
        dh = min(h[i_, j_], H[i, j] - H[i_, j_])
    return dh


@njit
def update_a(mh, mw, i, j, dh_left, dh_right, dh_up, dh_down, s, k, a):
    if 0 < i <= mh - 1:
        d = k * dh_up / s
        a[i-1, j] += d
        a[i, j] -= d
    if 0 <= i < mh - 1:
        d = k * dh_down / s
        a[i+1, j] += d
        a[i, j] -= d
    if 0 < j <= mw - 1:
        d = k * dh_left / s
        a[i, j-1] += d
        a[i, j] -= d
    if 0 <= j < mw - 1:
        d = k * dh_right / s
        a[i, j+1] += d
        a[i, j] -= d


@njit
def update(z, h, v, a, k, dt):
    mh, mw = z.shape
    H = z + h
    a[:, :] = 0.0

    for i in range(mh):
        for j in range(mw):
            if h[i, j] > 0:
                s, active_left, active_right, active_up, active_down = get_active(mh, mw, z, H, i, j)
                if s > 0:
                    dh_left = get_height_diff(active_left, H, h, i, j, 0, -1)
                    dh_right = get_height_diff(active_right, H, h, i, j, 0, 1)
                    dh_up = get_height_diff(active_up, H, h, i, j, -1, 0)
                    dh_down = get_height_diff(active_down, H, h, i, j, 1, 0)

                    update_a(mh, mw, i, j, dh_left, dh_right, dh_up, dh_down, s, k, a)

    v += dt * a
    h += dt * v



def plot(z, h):
    plt.imshow(z + h, vmin=0, vmax=4)
    plt.pause(1e-6)


def main():
    dim = 501
    z = np.zeros((dim, dim)) + 1.0
    h = np.zeros_like(z) + 1.0
    h[dim // 2, dim // 2] += 1
    v = np.zeros_like(h)
    a = np.zeros_like(h)

    k = 1.0
    dt = 0.1
    plt.figure(figsize=(4, 4))
    for _ in tqdm(range(10000)):
        # if _ % 10 == 0:
        #     plot(z, h)
        update(z, h, v, a, k, dt)

    plot(z, h)


if __name__ == "__main__":
    main()