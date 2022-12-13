import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider

from height_map import generate_height_map
from engine import FastErosionEngine


MAX_WATER_LEVEL = 1


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


def get_composite_image(z, h, u, v):
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

    m = 1.0
    clipped_water_depth = np.clip(h, 0, m) / m
    clipped_speed = np.clip(h * np.sqrt(u**2 + v**2), 0, 10) / 100

    sunlit = np.clip(cosine_similarity, -1, 1) * 32
    land_rgb = np.zeros((z.shape[0], z.shape[1], 3))
    land_rgb[:, :, 0] = 31 + sunlit / 3
    land_rgb[:, :, 1] = 95 + sunlit
    land_rgb[:, :, 2] = 31 + sunlit / 3

    water_rgb = np.zeros_like(land_rgb)
    water_rgb[:, :, 2] = 128

    turbulence_rgb = np.zeros_like(land_rgb) + 191
    turbulence_rgb[:, :, 2] = 255

    imgarr = np.multiply((1 - clipped_water_depth)[:, :, np.newaxis], land_rgb)
    imgarr += np.multiply(clipped_water_depth[:, :, np.newaxis], water_rgb)

    imgarr = np.multiply(1 - clipped_speed[:, :, np.newaxis], imgarr)
    imgarr += np.multiply(clipped_speed[:, :, np.newaxis], turbulence_rgb)

    imgarr = np.clip(imgarr, 0, 255).astype(np.uint8)
    return imgarr


def get_water_level_image(z, h):
    imgarr = (np.clip(h, 0, MAX_WATER_LEVEL) / MAX_WATER_LEVEL * 255).astype(np.uint8)
    return imgarr


def get_velocity_image(h, u, v):
    imgarr = np.stack(
        [np.clip(u * 10, 0, 255), h * 0, np.clip(v * 10, 0, 255)],
        axis=2,
    ).astype(np.uint8)
    return imgarr


def get_sediment_image(s):
    rgb = np.zeros((s.shape[0], s.shape[1], 3))
    rgb[:, :, 1] = np.clip(s * 1000, 0, 255)
    return rgb.astype(np.uint8)


def initialize_engine(seed, map_size, random_amplitude, slope_amplitude, smoothness, rainfall):
    z = generate_height_map(map_size, map_size, seed, smoothness) * random_amplitude
    z += np.linspace(0, 1, map_size).reshape(-1, 1).dot(np.ones(map_size).reshape(1, -1)) * slope_amplitude
    r = np.zeros_like(z)
    h = np.zeros_like(z)

    if seed > 0:
        r += rainfall
    else:
        r[map_size - 8, map_size // 2 - 4:map_size // 2 + 4] = rainfall

    engine = FastErosionEngine(
        z.astype(np.float32),
        h.astype(np.float32),
        r.astype(np.float32),
    )

    return engine
