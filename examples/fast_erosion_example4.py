import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.height_map import generate_height_map
from src.engine import FastErosionEngine


def generator():
    while True:
        yield


def main():
    # define height map
    z = generate_height_map(256, 256, 32)

    # define water height
    h = np.maximum(0, 25 - z)

    # define rainfall
    r = 0.01 + np.zeros_like(z)
    # r[100, 110] = 1

    engine = FastErosionEngine(z, h, r)
    dt = 0.1

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(dt, k_c=0.1)

        if i % 100 == 0:
            print(np.sum(engine.h))
            axes[0, 0].set_title("terrain height")
            axes[0, 0].imshow(engine.z, vmin=-128, vmax=128)

            axes[0, 1].set_title("sediment concentration")
            axes[0, 1].imshow(engine.s, vmin=0, vmax=0.1)

            axes[1, 0].set_title("water height")
            axes[1, 0].imshow(engine.h, vmin=0, vmax=50)

            axes[1, 1].set_title("velocity")
            axes[1, 1].imshow(np.sqrt(engine.u**2 + engine.v**2), vmin=0, vmax=1)

            plt.draw()
            plt.pause(0.0001)


        i += 1

if __name__ == "__main__":
    main()