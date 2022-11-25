import numpy as np
from numba import njit


# Simulation constants, mostly taken from original code
A_PIPE = 0.6  # virtual pipe cross section
G = 9.81      # gravitational acceleration
L_PIPE = 1    # virtual pipe length
LX = 1        # horizontal distance between grid points
LY = 1        # vertical distance between grid points

K_c = 0.1     # sediment capacity constant
K_s = 0.1     # dissolving constant
K_d = 0.1     # deposition constant
K_e = 0.01    # evaporation constant


@njit
def update(
    z,   # terrain height
    h,   # water height
    r,   # rainfall
    s,   # suspended sediment amount
    fL,  # flux towards left neighbor
    fR,  # flux towards right neighbor
    fT,  # flux towards top neighbor
    fB,  # flux towards bottom neighbor
    dt,  # time step
):
    n_x = z.shape[1]
    n_y = z.shape[0]

    H = z + h  # surface height

    # auxiliary arrays - should be buffers to speed up things?
    dvol = np.zeros_like(z)
    h1 = np.zeros_like(z)
    h2 = np.zeros_like(z)
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    s1 = np.zeros_like(z)

    #####################################################
    # the following section titles and equation numbers #
    # were taken from the original paper.               #
    # the implementation here differs somewhat from the #
    # original code.                                    #
    #####################################################

    # 3.1 Water Increment
    h1 = h + dt * r  # rainfall increment (eqn 1) TODO: include in loop

    # 3.2.1 outflow flux computation
    # we put this in a separate loop because we need all
    # fluxes calculated before proceeding with calculating
    # water and sediment transportation
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            # eqn 3
            # difference in height between tile (j, i) and its neighbors.
            # this drives the initial flux calculations
            dhL = H[j, i] - H[j, i - 1]
            dhR = H[j, i] - H[j, i + 1]
            dhT = H[j, i] - H[j - 1, i]
            dhB = H[j, i] - H[j + 1, i]

            # eqn 2
            # calculate flux from tile (j, i) to each neighbor.
            # we don't allow negative flux
            flux_factor = dt * A_PIPE / L_PIPE * G
            fL[j, i] = max(0, fL[j, i] + dhL * flux_factor)
            fR[j, i] = max(0, fR[j, i] + dhR * flux_factor)
            fT[j, i] = max(0, fT[j, i] + dhT * flux_factor)
            fB[j, i] = max(0, fB[j, i] + dhB * flux_factor)

            # eqn 4
            # calculate K.
            # K is an adjustment factor to make sure that
            # the outflow does not lead to a "negative"
            # water level in the tile.
            sum_f = fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]
            if sum_f > 0:
                K = min(
                    1,
                    h1[j, i] * LX * LY / (sum_f * dt),
                )
            else:
                K = 1

            # eqn 5
            fL[j, i] *= K
            fR[j, i] *= K
            fT[j, i] *= K
            fB[j, i] *= K

    # setting edge fluxes to 0 to prevent leaking
    fL[0, :] = 0
    fR[-1, :] = 0
    fT[:, 0] = 0
    fB[:, -1] = 0

    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            # 3.2.2 water surface and velocity field update

            # flux coming into (j, i)
            sum_f_in = (
                fR[j, i - 1] + fT[j + 1, i] + fL[j, i + 1] + fB[j - 1, i]
            )

            # flux going out of (j, i)
            sum_f_out = (
                fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]
            )

            # eqn 6: delta volume
            dvol[j, i] = dt * (sum_f_in - sum_f_out)

            # eqn 7: mean water level between rainfall and outflow
            h2[j, i] = h1[j, i] + dvol[j, i] / (LX * LY)
            h_mean = 0.5 * (h1[j, i] + h2[j, i])

            # eqn 8
            dwx = fR[j, i - 1] - fL[j, i] + fR[j, i] - fL[j, i + 1]
            dwy = fB[j - 1, i] - fT[j, i] + fB[j, i] - fT[j + 1, i]

            # eqn 9. note that we add a conditional,
            # otherwise we would be dividing by 0
            if h_mean > 0:
                u[j, i] = dwx / LY / h_mean
                v[j, i] = dwy / LX / h_mean
            else:
                u[j, i] = 0
                v[j, i] = 0

            # 3.3 erosion and deposition

            # first, calculate (approximate) gradient
            dhdy = 0.5 * (h2[j + 1, i] - h2[j - 1, i])
            dhdx = 0.5 * (h2[j, i + 1] - h2[j, i - 1])

            # dot product will give the grade (slope magnitude per unit length)
            # in the direction of the gradient
            dot_product = dhdx**2 + dhdy**2

            # this follows from Pythagoras
            sin_local_tilt = np.sqrt(dot_product / (dot_product + 1))

            # eqn 10:
            # calculate sediment transport capacity C
            # we use a minimum of 0.15 for the slope to keep things "interesting"
            C = K_c * max(0.15, sin_local_tilt) * np.sqrt(u[j, i] ** 2 + v[j, i] ** 2)

            if C > s[j, i]:
                # if capacity exceeds suspended sediment,
                # erode soil and add it to sediment
                delta_soil = K_s * (C - s[j, i])

                # eqn 11a
                z[j, i] -= delta_soil

                # eqn 11b
                s1[j, i] = s[j, i] + delta_soil
            else:
                # if suspended sediment exceeds capacity,
                # deposit sediment and substract it from sediment.
                # TODO: this can be probably be simplified so we don't need a conditional!
                delta_soil = K_d * (s[j, i] - C)

                # eqn 12a
                z[j, i] += delta_soil

                # eqn 12b
                s1[j, i] = s[j, i] - delta_soil

            # 3.4 sediment transportation
            # eqn 14: changed this a bit so we get a nearest neighbor
            # instead of proper interpolation
            j1 = int((j - dt * u[j, i]) // 1)
            i1 = int((i - dt * v[j, i]) // 1)

            j1 = max(0, min(j1, n_x))
            i1 = max(0, min(i1, n_y))

            s[j, i] = s1[j1, i1]

            # 3.5 evaporation
            h[j, i] = h2[j, i] * (1 - K_e * dt)

    return z, h, s, fL, fR, fT, fB, u, v
