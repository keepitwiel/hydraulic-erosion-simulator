import numpy as np
from matplotlib import pyplot as plt


def plot(axes, engine):
    axes[0, 0].set_title("green=z, blue=u")
    rgb = np.concatenate(
        [
            np.zeros_like(engine.z, dtype=np.uint8)[:, :, np.newaxis],
            np.clip(engine.z[:, :, np.newaxis], 0, 255).astype(np.uint8),
            (255 * np.clip(engine.h[:, :, np.newaxis], 0, 1)).astype(np.uint8),
        ],
        axis=2
    )
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_xlabel("u")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].set_title("Water height")
    axes[0, 1].imshow(engine.h, vmin=0, vmax=1)

    axes[1, 0].set_title("erosion")
    axes[1, 0].imshow(engine.z0 - engine.z, vmin=-10, vmax=10)

    axes[1, 1].set_title("z")
    axes[1, 1].imshow(engine.z, vmin=np.min(engine.z0), vmax=np.max(engine.z0))
    plt.draw()
    plt.pause(0.0001)