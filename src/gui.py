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
        self.seed = 42

        self.layout_main = QHBoxLayout()
        self.setLayout(self.layout_main)
        self.layout_left = QVBoxLayout()
        self.layout_right = QVBoxLayout()

        self._init_left()
        self._init_right()

        self.mode = "composite"

        self.z = None
        self.h = None

    def _init_left(self):
        # erode_button
        self.erode_button = QPushButton(text="Erode")
        self.erode_button.setMinimumHeight(100)
        self.erode_button.pressed.connect(self.erode_terrain)
        self.layout_left.addWidget(self.erode_button)
        self.layout_main.addLayout(self.layout_left, stretch=1)

        self.sliders = {}
        self.labels = {}
        self._add_slider("Map size", 7, 9, 1, 8, lambda x: 2**x)
        self._add_slider("Water level", -5, 5, 1, 0, lambda x: 10*x)
        self._add_slider("Rainfall", -4, 0, 1, -2, lambda x: 10**x)
        self._add_slider("Sediment capacity constant", -5, 0, 1, -5, lambda x: 10**x)
        self._add_slider("Number of iterations", 0, 10, 1, 1, lambda x: 100*x)

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
        self.update_image()

    def create_terrain(self):
        map_size = self.sliders["Map size"].mapped_value()
        seed = self.seed

        self.z0 = generate_height_map(map_size, map_size, seed) * 256
        self.update_water_level()

    def update_water_level(self):
        initial_water_level = self.sliders["Water level"].mapped_value()
        self.h0 = np.maximum(0, initial_water_level - self.z0)

    def update_image(self):
        if self.z is not None and self.h is not None:
            if self.mode == "composite":
                self.update_composite_image()
            elif self.mode == "water level":
                self.update_water_image()

    def update_composite_image(self):
        m = 10
        b = np.clip(self.h, 0, m) / m * 128

        land = np.zeros((self.z.shape[0], self.z.shape[1], 3))
        land[:, :, 1] = 128 - b

        water = np.zeros_like(land)
        water[:, :, 2] = b

        imgarr = (land + water).astype(np.uint8)
        self.img.setImage(imgarr)

    def update_water_image(self):
        m = 10
        b = (np.clip(self.h, 0, m) / m * 255).astype(np.uint8)
        self.img.setImage(b)

    def erode_terrain(self):
        dt = 0.1
        K_c = self.sliders["Sediment capacity constant"].mapped_value()
        self.create_terrain()
        self.update_water_level()
        rainfall = 10 ** self.sliders["Rainfall"].mapped_value()
        self.r0 = np.zeros_like(self.z0) + rainfall

        engine = FastErosionEngine(
            self.z0.astype(np.float32),
            self.h0.astype(np.float32),
            self.r0.astype(np.float32),
        )
        for _ in tqdm(range(self.sliders["Number of iterations"].mapped_value())):
            engine.update(dt, K_c)
        self.z = engine.z
        self.h = engine.h
        self.update_image()
        del engine


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(1280, 720)
    w.show()
    sys.exit(app.exec_())
