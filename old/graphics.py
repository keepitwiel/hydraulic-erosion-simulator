import numpy as np
from matplotlib import pyplot as plt


def plot(axes, engine):
    # axes[0, 0].set_title("green=z, blue=u")
    # rgb = np.concatenate(
    #     [
    #         np.zeros_like(engine.z, dtype=np.uint8)[:, :, np.newaxis],
    #         np.clip(engine.z[:, :, np.newaxis], 0, 255).astype(np.uint8),
    #         (255 * np.clip(engine.h[:, :, np.newaxis], 0, 1)).astype(np.uint8),
    #     ],
    #     axis=2
    # )
    # axes[0, 0].imshow(rgb)
    # axes[0, 0].set_xlabel("x")
    # axes[0, 0].set_ylabel("y")
    axes[0, 0].set_title("Surface height (z + h)")
    axes[0, 0].imshow(engine.z + engine.h)

    axes[0, 1].set_title("Water height (h)")
    axes[0, 1].imshow(engine.h, vmin=0, vmax=1)

    axes[1, 0].set_title("erosion (z0 - z)")
    axes[1, 0].imshow(engine.z0 - engine.z, vmin=-2, vmax=2)

    axes[1, 1].set_title("terrain height (z)")
    axes[1, 1].imshow(engine.z)
    plt.draw()
    plt.pause(0.0001)