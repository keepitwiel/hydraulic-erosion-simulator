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
    z = np.array(
        [
             [-50.0 + 0.1 * (y - 64)**2 + 1.0*x for x in range(128)] for y in range(128)
        ]
    )
    z += np.random.normal(0, 0.1, size=z.shape)

    # define water height
    # h = np.zeros_like(z)
    # h[60:68, 127] = 100.0
    h = np.maximum(0, -z)

    # define rainfall
    # r = np.zeros_like(z) + 0.0001
    # r = np.random.exponential(0.0001, size=z.shape)
    r = np.zeros_like(z)
    r[50:60, 127] = 0.1
    r[0, 100:105] = 0.05
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
