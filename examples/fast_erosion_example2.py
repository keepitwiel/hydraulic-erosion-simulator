import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.engine import FastErosionEngine


def generator():
    while True:
        yield


def main():
    # define height map
    z = np.dot(
        np.ones(128).reshape(-1, 1),
        np.linspace(0, 10, 128).reshape(1, -1),
    )

    z[0, :] = 200
    z[-1, :] = 200
    z[:, 0] = 200
    z[:, -1] = 200
    z[63, 63] = 0.1

    # define water height
    h = np.zeros_like(z)
    h[:, :16] = 1

    # define rainfall
    r = np.zeros_like(z)
    r[63, 125] = 1

    engine = FastErosionEngine(z, h, r)
    dt = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(dt, k_c=0.1)

        if i % 1000 == 0:
            axes[0, 0].imshow(engine.z, vmin=0, vmax=10)

            axes[0, 1].set_title("sediment height")
            axes[0, 1].imshow(engine.s)

            axes[1, 0].set_title("u")
            axes[1, 0].imshow(engine.u, vmin=-1, vmax=1)

            axes[1, 1].set_title("v")
            axes[1, 1].imshow(engine.v, vmin=-1, vmax=1)
            plt.draw()
            plt.pause(0.0001)

        i += 1

if __name__ == "__main__":
    main()
