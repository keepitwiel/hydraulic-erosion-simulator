import sys

from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget,
    QComboBox,
)
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm
from scipy.ndimage import laplace

from height_map import generate_height_map
from engine import FastErosionEngine
from gui_utils import Slider, get_composite_image, get_terrain_height_image, get_surface_height_image


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

        self.generate_world()

    def _init_left(self):
        self.sliders = {}
        self.labels = {}
        self._add_slider("Random seed", 0, 255, 1, 18, lambda x: x)
        self._add_slider("Map size", 7, 9, 1, 8, lambda x: 2**x)
        # self._add_slider("Water level", -5, 5, 1, 0, lambda x: 10*x)
        self._add_slider("Rainfall", -4, 0, 1, -1, lambda x: 10**x)
        # self._add_slider("Rainfall Height", 0, 5, 1, 5, lambda x: 25*x)
        self._add_slider("Sediment capacity constant", -5, 0, 1, -2, lambda x: 10**x)
        self._add_slider("Number of iterations", 0, 3, 1, 2, lambda x: 10**x)

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
        self.mode_box.addItem("terrain height")
        self.mode_box.addItem("surface height")
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
        del self.engine

        map_size = self.sliders["Map size"].mapped_value()
        seed = self.sliders["Random seed"].mapped_value()
        print(f"seed: {seed}")

        z = generate_height_map(map_size, map_size, seed) * 256

        h = np.zeros_like(z)

        rainfall = self.sliders["Rainfall"].mapped_value()
        r = np.zeros_like(z) + rainfall

        self.engine = FastErosionEngine(
            z.astype(np.float32),
            h.astype(np.float32),
            r.astype(np.float32),
        )

        self.update_images()
        self.display_image()

    def update_images(self):
        if self.engine is not None:
            self.update_composite_image()
            self.update_water_image()
            self.update_terrain_height_image()
            self.update_surface_height_image()

    def update_composite_image(self):
        imgarr = get_composite_image(self.engine.z, self.engine.h)
        self.images["composite"] = imgarr

    def update_water_image(self):
        m = 1
        h = (np.clip(self.engine.h, 0, m) / m * 255).astype(np.uint8)
        self.images["water level"] = h

    def update_terrain_height_image(self):
        imgarr = get_terrain_height_image(self.engine.z, self.engine.h)
        self.images["terrain height"] = imgarr

    def update_surface_height_image(self):
        imgarr = get_surface_height_image(self.engine.z, self.engine.h)
        self.images["surface height"] = imgarr

    def erode_terrain(self):
        dt = 0.1
        k_c = self.sliders["Sediment capacity constant"].mapped_value()
        n_iters = self.sliders["Number of iterations"].mapped_value()
        for _ in tqdm(range(n_iters)):
            self.engine.update(dt, k_c)
            self.engine.z += dt * 0.001 * self.engine.g * laplace(self.engine.z)

        self.update_images()
        self.display_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec_())
