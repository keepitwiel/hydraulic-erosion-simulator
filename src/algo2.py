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
def get_diff(active, H, h, v, i, j, di, dj):
    dh = 0.0
    dv = 0.0
    if active:
        i_ = i + di
        j_ = j + dj
        dh = min(h[i, j], H[i, j] - H[i_, j_])
        dv = v[i, j] - v[i_, j_]
    return dh, dv


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
def update(z, h, v, a_p, a_v, k, nu, dt):
    mh, mw = z.shape
    H = z + h
    a_p[:, :] = 0.0
    a_v[:, :] = 0.0

    for i in range(mh):
        for j in range(mw):
            if h[i, j] > 0:
                s, active_left, active_right, active_up, active_down = get_active(mh, mw, z, H, i, j)
                if s > 0:
                    dh_left, dv_left = get_diff(active_left, H, h, v, i, j, 0, -1)
                    dh_right, dv_right = get_diff(active_right, H, h, v, i, j, 0, 1)
                    dh_up, dv_up = get_diff(active_up, H, h, v, i, j, -1, 0)
                    dh_down, dv_down = get_diff(active_down, H, h, v, i, j, 1, 0)

                    update_a(mh, mw, i, j, dh_left, dh_right, dh_up, dh_down, s, k, a_p)
                    update_a(mh, mw, i, j, dv_left, dv_right, dv_up, dv_down, s, nu, a_v)

    v += dt * (a_p + a_v)
    h += dt * v



def plot(z, h):
    plt.title(f"Water mass: {np.sum(h):.2f}")
    plt.imshow(z + h, vmin=10, vmax=12)
    plt.pause(1e-6)


def main():
    dim = 501
    z = np.zeros((dim, dim)) + 1.0
    h = np.zeros_like(z) + 10.0
    h[240:260, 240:260] += 1.0
    v = np.zeros_like(h)
    a_p = np.zeros_like(h)
    a_v = np.zeros_like(h)

    k = 5.0
    nu = 1.0
    dt = 0.1
    plt.figure(figsize=(4, 4))
    for _ in tqdm(range(10000)):
        if _ % 100 == 0:
            plot(z, h)
        update(z, h, v, a_p, a_v, k, nu, dt)

    plot(z, h)


if __name__ == "__main__":
    main()