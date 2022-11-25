import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from heightmap_diffusion import generate_height_map
from fast_erosion_engine import FastErosionEngine


def generator():
    while True:
        yield


def main():
    # define height map
    base = np.dot(
        np.ones((128, 1)),
        np.linspace(0, 10, 128).reshape(1, -1),
    )
    z = np.maximum(base, np.rot90(base))
    z = np.maximum(z, np.rot90(z))
    z += np.random.exponential(0.001, size=z.shape)

    # define water height
    h = np.maximum(0, 7 - z)


    z[0, :] = 200
    z[-1, :] = 200
    z[:, 0] = 200
    z[:, -1] = 200


    # define rainfall
    r = 0.01 + np.zeros_like(z)
    # r[63, 125] = 1

    engine = FastErosionEngine(z, h, r)
    dt = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(dt)

        if i % 1000 == 0:
            print(np.sum(engine.h))
            axes[0, 0].set_title("terrain height")
            axes[0, 0].imshow(engine.z, vmin=0, vmax=12)

            axes[0, 1].set_title("sediment concentration")
            axes[0, 1].imshow(engine.s, vmin=0, vmax=0.1)

            axes[1, 0].set_title("water height")
            axes[1, 0].imshow(engine.h, vmin=0, vmax=2)

            axes[1, 1].set_title("velocity")
            axes[1, 1].imshow(np.sqrt(engine.u**2 + engine.v**2), vmin=0, vmax=1)

            plt.draw()
            plt.pause(0.0001)
            engine.z[1:-1, 1:-1] = gaussian_filter(engine.z[1:-1, 1:-1], sigma=0.1)

        i += 1

if __name__ == "__main__":
    main()