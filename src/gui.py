import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget,
    QComboBox,
)
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from height_map import generate_height_map
from engine import FastErosionEngine
from gui_utils import Slider


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
        self.z0 = None
        self.h0 = None
        self.r0 = None

        self.layout_main = QHBoxLayout()
        self.setLayout(self.layout_main)
        self.layout_left = QVBoxLayout()
        self.layout_right = QVBoxLayout()

        self._init_left()
        self._init_right()

        self.mode = "composite"
        self.engine = None
        self.images = {}

    def _init_left(self):
        self.sliders = {}
        self.labels = {}
        self._add_slider("Random seed", 0, 255, 1, 42, lambda x: x)
        self._add_slider("Map size", 7, 9, 1, 8, lambda x: 2**x)
        self._add_slider("Water level", -5, 5, 1, 0, lambda x: 10*x)
        self._add_slider("Rainfall", -4, 0, 1, -2, lambda x: 10**x)
        self._add_slider("Sediment capacity constant", -5, 0, 1, -5, lambda x: 10**x)
        self._add_slider("Number of iterations", 0, 10, 1, 1, lambda x: 100*x)

        # reset button
        self.reset_button = QPushButton(text="Generate World")
        self.reset_button.setMinimumHeight(100)
        self.reset_button.pressed.connect(self.generate_world)
        self.layout_left.addWidget(self.reset_button)

        # erode_button
        self.erode_button = QPushButton(text=f"Run simulation {self.sliders['Number of iterations'].mapped_value()} steps")
        self.erode_button.setMinimumHeight(100)
        self.erode_button.pressed.connect(self.erode_terrain)
        self.layout_left.addWidget(self.erode_button)
        self.layout_main.addLayout(self.layout_left, stretch=1)

    def _init_right(self):
        # image mode dropdown
        self.mode_box = QComboBox()
        self.mode_box.setMaximumWidth(200)
        self.mode_box.addItem("composite")
        self.mode_box.addItem("water level")
        self.mode_box.addItem("water velocity")
        self.mode_box.addItem("terrain height")
        self.mode_box.textActivated.connect(self.set_mode)
        self.layout_right.addWidget(self.mode_box)

        # image
        self.img = pg.RawImageWidget()
        arr = np.zeros(shape=(128, 128), dtype=np.uint8)
        self.img.setImage(arr)
        self.layout_right.addWidget(self.img)

        self.layout_main.addLayout(self.layout_right, stretch=3)

    def _add_slider(self, name, minimum, maximum, step_size, initial, value_mapping):
        self.labels[name] = QLabel(f"{name} [{value_mapping(minimum)} ... {value_mapping(maximum)}]")
        self.labels[name].setMaximumHeight(20)
        self.layout_left.addWidget(self.labels[name])
        self.sliders[name] = Slider(minimum, maximum, step_size, initial, value_mapping)
        self.layout_left.addWidget(self.sliders[name])

    def set_mode(self, item):
        self.mode = item
        self.display_image()

    def display_image(self):
        if self.mode in self.images.keys():
            self.img.setImage(self.images[self.mode])
        else:
            print(f"Image mode {self.mode} not found!")

    def generate_world(self):
        """
        Generate world and store it inside the engine.

        :return: None
        """
        map_size = self.sliders["Map size"].mapped_value()
        seed = self.sliders["Random seed"].mapped_value()

        z = generate_height_map(map_size, map_size, seed) * 256

        initial_water_level = self.sliders["Water level"].mapped_value()
        h = np.maximum(0, initial_water_level - z)

        rainfall = self.sliders["Rainfall"].mapped_value()
        r = np.zeros_like(z) + rainfall

        self.engine = FastErosionEngine(
            z.astype(np.float32),
            h.astype(np.float32),
            r.astype(np.float32),
        )

        self.update_images()
        self.display_image()

    # def update_water_level(self):
    #     initial_water_level = self.sliders["Water level"].mapped_value()
    #     self.h0 = np.maximum(0, initial_water_level - self.z0)

    def update_images(self):
        if self.engine is not None:
            self.update_composite_image()
            self.update_water_image()

    def update_composite_image(self):
        h = self.engine.h
        z = self.engine.z

        grad = np.gradient(z)
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
        b = np.clip(h, 0, m) / m * 128

        sunlit = np.clip(cosine_similarity, -1, 1) * 32 * (b < 0.1)
        land = np.zeros((z.shape[0], z.shape[1], 3))
        land[:, :, 1] = 128 - b + sunlit

        water = np.zeros_like(land)
        water[:, :, 2] = b

        imgarr = (land + water).astype(np.uint8)
        self.images["composite"] = imgarr

    def update_water_image(self):
        m = 10
        h = (np.clip(self.engine.h, 0, m) / m * 255).astype(np.uint8)
        self.images["water level"] = h

    def erode_terrain(self):
        dt = 0.1
        k_c = self.sliders["Sediment capacity constant"].mapped_value()
        n_iters = self.sliders["Number of iterations"].mapped_value()
        for _ in tqdm(range(n_iters)):
            self.engine.update(dt, k_c)
        self.update_images()
        self.display_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(1280, 720)
    w.show()
    sys.exit(app.exec_())
