import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

DOMAIN_WIDTH = 200
DOMAIN_HEIGHT = 200
FIGURE_SIZE = (4, 4)

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = -0.9
CONSTANT_FORCE = np.array([0.0, -0.1])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 5000

PLOT_EVERY = 50
SCATTER_DOT_SIZE = 10

DOMAIN_X_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH])
DOMAIN_Y_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH])

NORMALIZATION_DENSITY = (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9)
NORMALIZATION_PRESSURE_FORCE = -(45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)
NORMALIZATION_VISCOUS_FORCE = (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)


def plot(positions):
    plt.scatter(
        positions[:, 0],
        positions[:, 1],
        s=SCATTER_DOT_SIZE,
        c=positions[:, 1],
        cmap="Wistia_r",
    )
    plt.ylim(DOMAIN_Y_LIM)
    plt.xlim(DOMAIN_X_LIM)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    plt.clf()


@njit
def enforce_boundary_conditions(x, xdot):
    out_of_left_boundary = x[:, 0] < DOMAIN_X_LIM[0]
    out_of_right_boundary = x[:, 0] > DOMAIN_X_LIM[1]
    out_of_bottom_boundary = x[:, 1] < DOMAIN_Y_LIM[0]
    out_of_top_boundary = x[:, 1] > DOMAIN_Y_LIM[1]

    xdot[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
    x[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

    xdot[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
    x[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

    xdot[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
    x[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

    xdot[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
    x[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

    return x, xdot


@njit
def calculate_force(n_particles, x, xdot, d, rho, p, f):
    for i in range(n_particles):
        # calc density
        for j in range(n_particles):
            if d[i, j] < SMOOTHING_LENGTH:
                rho[i] += NORMALIZATION_DENSITY * (
                        SMOOTHING_LENGTH ** 2 - d[i, j] ** 2
                ) ** 3

        # calc pressure
        p[i] = ISOTROPIC_EXPONENT * (rho[i] - BASE_DENSITY)

    # calc forces between particles
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                u = SMOOTHING_LENGTH - d[i, j]
                if u > 0:
                    # pressure force
                    f[i] += NORMALIZATION_PRESSURE_FORCE * (
                        -(x[j] - x[i])
                    ) / d[i, j] * (
                        p[j] + p[i]
                    ) / (
                            2 * rho[j]
                    ) * u ** 2

                    # viscous force
                    f[i] += NORMALIZATION_VISCOUS_FORCE * (
                        xdot[j] - xdot[i]
                    ) / rho[j] * u

        # gravitational force
        f[i] += CONSTANT_FORCE

    return f


def update(x, xdot):
    """
    We want to update
    x_(t+1) = x_t + delta * xdot_t
    xdot_(t+1) = xdot_t + delta * g(x_t, xdot_t)

    What is g?

    (1) g = (f_p(x_t, xdot_t) + f_v(x_t, xdot_t) + f_g) / rho(x_t)

    (2a) f'_p = f_p / rho
    (2b) f'_v = f_v / rho
    (2c) f'_g = f_g / rho

    (3) f'_p[i] = SUM_OVER_j[
            NORMALIZATION_PRESSURE_FORCE * (
            -(x_t[j] - x_t[i]) / d
            * (p_t[j] + p_t[i]) / (2 * rho_t[j])
            * (SMOOTHING_LENGTH - d) ** 2
        ) / rho[i]
    ]

    :param x:
    :param xdot:
    :return:
    """
    n = len(x)
    d = squareform(pdist(x))

    rho = np.zeros(n)
    p = np.zeros_like(rho)
    f = np.zeros_like(x)

    # calculate force
    f = calculate_force(n, x, xdot, d, rho, p, f)

    # increment position and velocity
    x += TIME_STEP_LENGTH * xdot
    xdot += TIME_STEP_LENGTH * f / rho[:, np.newaxis]

    # Enforce Boundary Conditions
    x, xdot = enforce_boundary_conditions(x, xdot)

    return x, xdot


def main():
    pos_x, pos_y = np.meshgrid(
        np.linspace(0.05 * DOMAIN_WIDTH, 0.45 * DOMAIN_WIDTH, 20),
        np.linspace(0.55 * DOMAIN_HEIGHT, 0.95 * DOMAIN_HEIGHT, 20))
    x = np.concatenate([pos_x.reshape(-1, 1), pos_y.reshape(-1, 1)], axis=1)  # np.zeros((n_particles, 2))
    x += np.random.normal(size=x.shape)
    xdot = np.zeros_like(x)

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    def generator():
        while True:
            yield

    i = 0
    for _ in tqdm(generator()):
        x, xdot = update(x, xdot)
        if i % PLOT_EVERY == 0:
            plot(x)
        i += 1


if __name__ == "__main__":
    main()
