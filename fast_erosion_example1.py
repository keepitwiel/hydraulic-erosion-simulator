import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from heightmap_diffusion import generate_height_map
from fast_erosion import FastErosionEngine


def generator():
    while True:
        yield


def main():
    # define height map
    z = generate_height_map(128, 128, 32)
    z[0, :] = 200
    z[-1, :] = 200
    z[:, 0] = 200
    z[:, -1] = 200

    # define water height
    h = np.zeros_like(z)

    # define rainfall
    r = np.zeros_like(z)
    r[100, 110] = 1

    engine = FastErosionEngine(z, h, r)
    dt = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(dt)

        if i % 1000 == 0:
            # print(f"iteration {i} done. water: {np.sum(h)}")
            axes[0, 0].set_title("green=z, blue=h")
            rgb = np.concatenate(
                [
                    np.zeros_like(engine.z, dtype=np.uint8)[:, :, np.newaxis],
                    np.clip(engine.z[:, :, np.newaxis], 0, 255).astype(np.uint8),
                    (8 * np.clip(engine.h[:, :, np.newaxis], 0, 16)).astype(np.uint8),
                ],
                axis=2
            )
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")

            axes[0, 1].set_title("Water height")
            axes[0, 1].imshow(engine.h)

            axes[1, 0].set_title("u")
            axes[1, 0].imshow(engine.u, vmin=-1, vmax=1)

            axes[1, 1].set_title("v")
            axes[1, 1].imshow(engine.v, vmin=-1, vmax=1)
            plt.draw()
            plt.pause(0.0001)

        i += 1

if __name__ == "__main__":
    main()