import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

DOMAIN_WIDTH = 500
DOMAIN_HEIGHT = 500
FIGURE_SIZE = (4, 4)

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = -0.9
CONSTANT_FORCE = 0.1
G = 0.02

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 5000

PLOT_EVERY = 100
SCATTER_DOT_SIZE = 1

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
    plt.ylim(-DOMAIN_HEIGHT, DOMAIN_HEIGHT)
    plt.xlim(-DOMAIN_WIDTH, DOMAIN_WIDTH)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    plt.clf()


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

    # calc forces between particles and update x, xdot
    for i in range(n_particles):
        for j in range(n_particles):
            if i != j:
                # gravity
                f[i] += (x[j] - x[i]) * G / (d[i, j] ** 2)

                # other forces
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

    return f, rho


def update(x, xdot):
    n = len(x)

    rho = np.zeros(n)
    p = np.zeros_like(rho)
    f = np.zeros_like(x)

    # calculate distance
    d = squareform(pdist(x))

    # calculate force
    f, rho = calculate_force(n, x, xdot, d, rho, p, f)

    # increment position and velocity
    delta_x = TIME_STEP_LENGTH * xdot
    delta_xdot = TIME_STEP_LENGTH * f / rho[:, np.newaxis]

    # active = np.sum(np.abs(delta_x) > 0.01)
    # print(active)

    x += delta_x
    xdot += delta_xdot

    return x, xdot


def main():
    pos_x, pos_y = np.meshgrid(
        2 * np.array([-100, -95, -90, -85, -80, 70, 75, 80, 85, 90], dtype=float),
        2 * np.array([-100, -95, -90, -85, -80, 70, 75, 80, 85, 90], dtype=float),
        # np.linspace(-100, 100, 10),
        # np.linspace(-100, 100, 10)
    )
    x = np.concatenate([pos_x.reshape(-1, 1), pos_y.reshape(-1, 1)], axis=1)
    x += np.random.normal(0, 1, size=x.shape)
    xdot = x[:, ::-1] * 0.03 * np.array([-1, 1])

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
