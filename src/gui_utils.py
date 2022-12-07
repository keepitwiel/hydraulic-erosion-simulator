import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider


class Slider(QSlider):
    def __init__(
        self,
        minimum: int,
        maximum: int,
        step_size: int,
        initial: int,
        value_mapping: callable,
    ):
        super().__init__(Qt.Horizontal)
        self.setMaximumHeight(20)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setTickInterval(step_size)
        self.setSingleStep(step_size)
        self.setValue(initial)
        self.setSliderPosition(initial)
        self.setTickPosition(QSlider.TicksBelow)
        self.value_mapping = value_mapping

    def mapped_value(self):
        return self.value_mapping(self.value())


def get_terrain_height_image(z, h):
    altitude = (np.clip(z, -256, 255) // 16) * 16
    rgb = np.zeros((z.shape[0], z.shape[1], 3))
    rgb[:, :, 0] = altitude
    rgb[:, :, 1] = altitude
    rgb[:, :, 2] = altitude
    return rgb.astype(np.uint8)


def get_surface_height_image(z, h):
    altitude = (np.clip(z + h, -256, 255) // 16) * 16
    rgb = np.zeros((z.shape[0], z.shape[1], 3))
    rgb[:, :, 0] = altitude
    rgb[:, :, 1] = altitude
    rgb[:, :, 2] = altitude
    return rgb.astype(np.uint8)


def get_composite_image(z, h):
    # cosine similarity: determines shade
    grad = np.clip(np.gradient(z), -10, 10)
    theta = np.zeros_like(grad)
    theta[0, :, :] = -1/np.sqrt(2)
    theta[1, :, :] = -1/np.sqrt(2)

    cosine_similarity = -(
        np.multiply(
            grad[0], theta[0, :, :]
        ) + np.multiply(
            grad[1], theta[1, :, :]
        )
    ) / (
        np.sqrt(np.multiply(grad[0], grad[0]) + np.multiply(grad[1], grad[1]))
    )

    m = 10
    b = np.clip(h, 0, m) / m

    sunlit = np.clip(cosine_similarity, -1, 1) * 32 # * (b < 0.1)
    land = np.zeros((z.shape[0], z.shape[1], 3))
    land[:, :, 1] = (128 + sunlit) * (1 - b)

    water = np.zeros_like(land)
    water[:, :, 2] = 128 * b

    imgarr = (land + water).astype(np.uint8)
    return imgarr
