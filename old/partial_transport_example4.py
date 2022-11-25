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
    z = generate_height_map(128, 128, 42, smoothness=0.1) / 10 + 0.5

    # define water height
    h = np.maximum(0, -z)

    # define rainfall
    r = np.zeros_like(z)
    r[96:100, 96:100] = 0.001
    z[96:100, 96:100] += 100  # erosion "budget"
    # r[0, 100:105] = 0.05
    # define engine
    engine = PartialTransportEngine(z, h, k=0.1)

    # define output plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    i = 0
    for _ in tqdm(generator()):
        engine.update(r)
        if i % 1000 == 0:
            plot(axes, engine)
        i += 1


if __name__ == "__main__":
    main()
