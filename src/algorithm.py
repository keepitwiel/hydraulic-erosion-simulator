import numpy as np
from numba import njit


# Simulation constants, mostly taken from original code
A_PIPE = 0.6  # virtual pipe cross section
G = 9.81      # gravitational acceleration
L_PIPE = 1    # virtual pipe length
LX = 1        # horizontal distance between grid points
LY = 1        # vertical distance between grid points

K_C = 0.1     # sediment capacity constant
K_S = 0.1     # dissolving constant
K_D = 0.1     # deposition constant
K_E = 0.01    # evaporation constant


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
    k_c = K_C, # sediment capacity constant
):
    n_x = z.shape[1]
    n_y = z.shape[0]

    H = z + h  # surface height

    # auxiliary arrays - should be buffers to speed up things?
    h2 = np.zeros_like(z)
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    s1 = np.zeros_like(z)

    #####################################################
    # the following section titles and equation numbers #
    # were taken from the original paper.               #
    # the implementation here differs somewhat from the #
    # original code.                                    #
    # ------------------------------------------------- #
    # Practical notes:                                  #
    # 1) We follow numpy matrix coordinate convention.  #
    # j = horizontal coordinate. low/high = left/right  #
    # i = vertical coordinate. low/high = top/bottom    #
    #                                                   #
    # 2) We don't update the edge tiles for most        #
    # fields.                                           #
    #####################################################

    # 3.1 Water Increment
    # ========================================================================
    h1 = h + dt * r  # rainfall increment (eqn 1)

    # we put this in a separate loop because we need all
    # fluxes calculated before proceeding with calculating
    # water and sediment transportation
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):

            # 3.2.1 outflow flux computation
            # ================================================================
            # eqn 3
            # ----------------------------------------------------------------
            # difference in height between tile (j, i) and its neighbors.
            # this drives the initial flux calculations
            dhL = H[j, i] - H[j, i - 1]
            dhR = H[j, i] - H[j, i + 1]
            dhT = H[j, i] - H[j - 1, i]
            dhB = H[j, i] - H[j + 1, i]

            # eqn 2
            # ----------------------------------------------------------------
            # update flux from tile (j, i) to each neighbor.
            # we don't allow negative flux
            flux_factor = dt * A_PIPE / L_PIPE * G
            fL[j, i] = max(0, fL[j, i] + dhL * flux_factor)
            fR[j, i] = max(0, fR[j, i] + dhR * flux_factor)
            fT[j, i] = max(0, fT[j, i] + dhT * flux_factor)
            fB[j, i] = max(0, fB[j, i] + dhB * flux_factor)

            # eqn 4
            # ----------------------------------------------------------------
            # calculate adjustment factor.
            # this is to make sure that the outflow does not lead to a
            # negative water level in the tile.
            sum_f = fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]

            if sum_f > 0:
                adjustment_factor = min(1, h1[j, i] * LX * LY / (sum_f * dt))

                # eqn 5. We only need to calculate this step when sum_f > 0
                # ------------------------------------------------------------
                fL[j, i] *= adjustment_factor
                fR[j, i] *= adjustment_factor
                fT[j, i] *= adjustment_factor
                fB[j, i] *= adjustment_factor

    # setting edge fluxes to 0 to prevent leaking.
    # TODO: find out if this needs to be done before eqn 5!
    fL[0, :] = 0
    fR[-1, :] = 0
    fT[:, 0] = 0
    fB[:, -1] = 0

    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            # 3.2.2 water surface and velocity field update
            # ================================================================

            # flux coming into (j, i)
            sum_f_in = (
                fR[j, i - 1] + fT[j + 1, i] + fL[j, i + 1] + fB[j - 1, i]
            )

            # flux going out of (j, i)
            sum_f_out = (
                fL[j, i] + fR[j, i] + fT[j, i] + fB[j, i]
            )

            # eqn 6: delta volume
            # ----------------------------------------------------------------
            dvol = dt * (sum_f_in - sum_f_out)

            # eqn 7: update water height
            # ----------------------------------------------------------------
            dh = dvol / (LX * LY)
            h2[j, i] = h1[j, i] + dh

            # mean water level between rainfall and outflow.
            # TODO: do we really need mean? can't we just use h2?
            h_mean = h1[j, i] + 0.5 * dh

            if h_mean > 0:
                # we only calculate velocity if there is water, otherwise
                # velocity should be 0

                # eqn 8
                # ----------------------------------------------------------------
                dwx = fR[j, i - 1] - fL[j, i] + fR[j, i] - fL[j, i + 1]
                dwy = fB[j - 1, i] - fT[j, i] + fB[j, i] - fT[j + 1, i]

                # eqn 9
                # ----------------------------------------------------------------
                u[j, i] = dwx / LY / h_mean
                v[j, i] = dwy / LX / h_mean
            else:
                u[j, i] = 0
                v[j, i] = 0

            # 3.3 erosion and deposition
            # ================================================================

            # first, calculate (approximate) gradient
            dhdy = 0.5 * (h2[j + 1, i] - h2[j - 1, i])
            dhdx = 0.5 * (h2[j, i + 1] - h2[j, i - 1])

            # dot product will give the grade (slope magnitude per unit length)
            # in the direction of the gradient
            dot_product = dhdx**2 + dhdy**2

            # Now we want to calculate the sine of the local tilt angle.
            # this follows from Pythagoras:
            sin_local_tilt = np.sqrt(dot_product / (dot_product + 1))

            # eqn 10
            # ----------------------------------------------------------------
            # calculate sediment transport capacity C
            # we use a minimum of 0.15 for the slope to keep things "interesting"
            C = k_c * max(0.15, sin_local_tilt) * np.sqrt(u[j, i] ** 2 + v[j, i] ** 2)

            if C > s[j, i]:
                # if capacity exceeds suspended sediment,
                # erode soil and add it to sediment
                delta_soil = K_S * (C - s[j, i])

                # eqn 11a
                z[j, i] -= delta_soil

                # eqn 11b
                s1[j, i] = s[j, i] + delta_soil
            else:
                # if suspended sediment exceeds capacity,
                # deposit sediment and substract it from sediment.
                # TODO: this can be probably be simplified so we don't need a conditional!
                # -> would only work if K_S == K_D
                delta_soil = K_D * (s[j, i] - C)

                # eqn 12a
                z[j, i] += delta_soil

                # eqn 12b
                s1[j, i] = s[j, i] - delta_soil

            # 3.4 sediment transportation
            # ================================================================

            # now that we've absorbed or deposited the suspended sediment,
            # we can transport it.

            # eqn 14
            # ----------------------------------------------------------------
            # in short, s[j, i] = s1[j - u[j, i] * dt, i - v[j, i] * dt]

            # calculate coordinates "from where the sediment is coming from".
            # the sediment at that coordinate is the new value for
            # sediment at (j, i).
            j1 = j - dt * u[j, i]
            i1 = i - dt * v[j, i]

            # because j1 and i1 are not integer, we need to interpolate.
            # to do so, we calculate weights from j1, i1 to nearest neighbors.
            #
            #  j_lb, --------------- j_lb,
            #  i_lb                  i_ub
            #   |                     |
            #   |                     |
            #   |   x     j1, i1      |
            #   | ----- o             |
            #   |       |             |
            #   |       |             |
            #   |       | y           |
            #   |       |             |
            #  j_ub,    |            j_ub,
            #  i_lb  --------------- i_ub

            # calculte corner coordinates
            j_lb = int(j1)
            j_ub = j_lb + 1
            i_lb = int(i1)
            i_ub = i_lb + 1

            # calculate coordinates for interpolation
            x = j1 % 1
            y = i1 % 1

            # simple bilinear interpolation
            s[j, i] = (
                s1[j_lb, i_lb] * (1 - x) * (1 - y) +
                s1[j_ub, i_lb] * x * (1 - y) +
                s1[j_lb, i_ub] * (1 - x) * y +
                s1[j_ub, i_ub] * x * y
            )

            # 3.5 evaporation
            # ================================================================
            # eqn 15
            # ----------------------------------------------------------------
            h[j, i] = h2[j, i] * (1 - K_E * dt)

            # Not in original paper, it is however present in the code:
            # 3.6 Heuristic to remove sharp peaks/valleys
            # ================================================================
            # ... TODO: implement

    return z, h, s, fL, fR, fT, fB, u, v
