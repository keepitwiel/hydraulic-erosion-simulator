import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from heightmap_diffusion import generate_height_map
from partial_transport_engine import PartialTransportEngine
from graphics import plot


def generator():
    while True:
        yield


def main():
    # define height map
    z = np.dot(
        np.ones(128).reshape(-1, 1),
        np.linspace(0, 1, 128).reshape(1, -1),
    )
    z[:, 0:4] = -100
    # z[32:96:4, 32:96:4] = np.random.poisson(10, size=(16, 16))

    # define water height
    h = np.maximum(0, -z)

    ## define rainfall
    # r = np.zeros_like(z)
    # r[63, 127] = 1
    # z[63, 127] = 100
    # z[63, 63] += 1

    # define engine
    engine = PartialTransportEngine(z, h, k=0.1)

    # define output plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        r = np.zeros_like(z)
        if i % 10 == 0:
            r[:, 96:128] = np.random.poisson(0.01, size=(128, 32)) * 0.1
        engine.update(r)
        if i % 1000 == 0:
            plot(axes, engine)
        i += 1


if __name__ == "__main__":
    main()
