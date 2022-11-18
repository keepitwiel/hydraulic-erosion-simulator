import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from heightmap_diffusion import generate_height_map
from shaker_engine import ShakerEngine
from graphics import plot


def generator():
    while True:
        yield


def main():
    # define height map
    z = generate_height_map(128, 128, 32)

    # define water height
    h = np.zeros_like(z)

    # define rainfall
    r = np.zeros_like(z)
    r[32::64, 32::64] = 1

    # define engine
    engine = ShakerEngine(z, h, k=0.1)

    # define output plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(r)
        if i % 100 == 0:
            plot(axes, engine)
        i += 1


if __name__ == "__main__":
    main()
