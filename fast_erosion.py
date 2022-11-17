# based on https://hal.inria.fr/inria-00402079/document

import numpy as np
from numba import njit


@njit
def update(
    z,  # terrain height
    h,  # water height
    r,  # rainfall
    s,  # suspended sediment amount
    fL,  # flux towards left neighbor
    fR,  # flux towards right neighbor
    fT,  # flux towards top neighbor
    fB,  # flux towards bottom neighbor
    dt,  # time step
):
    n_x = z.shape[1]
    n_y = z.shape[0]
    
    # left -> right: x small -> large
    # top -> bottom: y small -> large
    
    A = 1 # A = virtual pipe cross section
    g = 9.81 # g = gravitational acceleration
    l = 1 # l = virtual pipe length
    lx = 1 # horizontal distance between grid points
    ly = 1 # vertical distance between grid points
    H = z + h   # surface height

    # auxiliary arrays
    dvol = np.zeros_like(z)
    h2 = np.zeros_like(z)
    u = np.zeros_like(z)
    v = np.zeros_like(z)

    # 3.1 Water Increment
    h1 = h + dt * r  # rainfall increment (eqn 1)

    # 3.2.1 outflow flux computation
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):

            # eqn 3
            dhL = H[j, i] - H[j, i - 1]
            dhR = H[j, i] - H[j, i + 1]
            dhT = H[j, i] - H[j - 1, i]
            dhB = H[j, i] - H[j + 1, i]

            # eqn 2
            fL[j, i] = max(0, fL[j, i] + dt * A * g * dhL / l)
            fR[j, i] = max(0, fR[j, i] + dt * A * g * dhR / l)
            fT[j, i] = max(0, fT[j, i] + dt * A * g * dhT / l)
            fB[j, i] = max(0, fB[j, i] + dt * A * g * dhB / l)

            # eqn 4
            sum_f = fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]
            if sum_f > 0:
                K = min(
                    1,
                    h1[j, i] * lx * ly / sum_f * dt,
                )
            else:
                K = 1

            # eqn 5
            fL[j, i] *= K
            fR[j, i] *= K
            fT[j, i] *= K
            fB[j, i] *= K

    # # disabled because it leaks water somehow
    # fL[0, :] = 0
    # fR[-1, :] = 0
    # fT[:, 0] = 0
    # fB[:, -1] = 0

    # 3.2.2 water surface and velocity field update
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            sum_f_in = (
                fR[j, i - 1] + fT[j + 1, i] + fL[j, i + 1] + fB[j - 1, i]
            )
            sum_f_out = (
                fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]
            )

            # eqn 6
            dvol[j, i] = dt * (sum_f_in - sum_f_out)

            # eqn 7
            h2[j, i] = h1[j, i] + dvol[j, i] / (lx * ly)
            h_mean = 0.5 * (h1[j, i] + h2[j, i])

            # eqn 8
            dwx = fR[j, i - 1] - fL[j, i] + fR[j, i] - fL[j, i + 1]

            # eqn 9
            if h_mean > 0:
                u[j, i] = dwx / ly / (h_mean)
            else:
                u[j, i] = 0

            # repeat for v
            dwy = fB[j - 1, i] - fT[j, i] + fB[j, i] - fT[j + 1, i]
            if h_mean > 0:
                v[j, i] = dwy / lx / (h_mean)
            else:
                v[j, i] = 0



    # 3.3 erosion and deposition
    # TODO

    # 3.4
    # TODO

    h = h2

    return z, h, s, fL, fR, fT, fB, u, v


class FastErosionEngine:
    def __init__(self, z0, h0, r0):
        self.z = z0
        self.h = h0
        self.r = r0

        self.s = np.zeros_like(z0)

        self.fL = np.zeros_like(z0)
        self.fR = np.zeros_like(z0)
        self.fT = np.zeros_like(z0)
        self.fB = np.zeros_like(z0)

        self.u = np.zeros_like(z0)
        self.v = np.zeros_like(z0)

    def update(self, dt):
        (
            self.z, self.h, self.s,
            self.fL, self.fR, self.fT, self.fB,
            self.u, self.v
        ) = update(
            self.z, self.h, self.r, self.s,
            self.fL, self.fR, self.fT, self.fB,
            dt,
        )
