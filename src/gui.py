import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton, QWidget,
    QComboBox,
)
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from height_map import generate_height_map
from engine import FastErosionEngine


class Slider(QSlider):
    def __init__(self, alignment, minimum, maximum, tick_interval, single_step, value):
        super().__init__(alignment)
        self.setMaximumHeight(20)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setTickInterval(tick_interval)
        self.setSingleStep(single_step)
        self.setValue(value)


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

    def _init_left(self):
        # erode_button
        self.erode_button = QPushButton(text="Erode")
        self.erode_button.setMinimumHeight(100)
        self.erode_button.pressed.connect(self.erode_terrain)
        self.layout_left.addWidget(self.erode_button)

        # map size slider
        map_size_label = QLabel("Map size")
        map_size_label.setMaximumHeight(20)
        self.layout_left.addWidget(map_size_label)
        self.map_size_slider = Slider(Qt.Horizontal, 5, 9, 1, 1, 7)
        self.layout_left.addWidget(self.map_size_slider)

        # water level slider
        water_level_label = QLabel("Water level")
        water_level_label.setMaximumHeight(20)
        self.layout_left.addWidget(water_level_label)
        self.water_level_slider = Slider(Qt.Horizontal, -50, 50, 1, 1, 0)
        self.layout_left.addWidget(self.water_level_slider)

        # rainfall slider
        rainfall_label = QLabel("Rainfall")
        rainfall_label.setMaximumHeight(20)
        self.layout_left.addWidget(rainfall_label)
        self.rainfall_slider = Slider(Qt.Horizontal, -6, 0, 1, 1, -6)
        self.layout_left.addWidget(self.rainfall_slider)

        # sediment capacity slider
        sediment_label = QLabel("Sediment capacity constant")
        sediment_label.setMaximumHeight(20)
        self.layout_left.addWidget(sediment_label)
        self.sediment_slider = Slider(Qt.Horizontal, 0, 10, 2, 2, 0)
        self.layout_left.addWidget(self.sediment_slider)

        # iterations slider
        iterations_label = QLabel("Number of iterations")
        iterations_label.setMaximumHeight(20)
        self.layout_left.addWidget(iterations_label)
        self.iterations_slider = Slider(Qt.Horizontal, 0, 1000, 100, 100, 100)
        self.layout_left.addWidget(self.iterations_slider)

        self.layout_main.addLayout(self.layout_left, stretch=1)

    def _init_right(self):
        # image mode dropdown
        self.mode_box = QComboBox()
        self.mode_box.setMaximumWidth(200)
        self.mode_box.addItem("composite")
        self.mode_box.addItem("water level")
        self.mode_box.addItem("water velocity")
        self.mode_box.addItem("terrain height")
        self.layout_right.addWidget(self.mode_box)

        # image
        self.img = pg.RawImageWidget()
        arr = np.zeros(shape=(128, 128), dtype=np.uint8)
        self.img.setImage(arr)
        self.layout_right.addWidget(self.img)

        self.layout_main.addLayout(self.layout_right, stretch=3)

    def create_terrain(self):
        map_size = 2 ** self.map_size_slider.value()
        seed = self.seed

        self.z0 = generate_height_map(map_size, map_size, seed) * 256
        self.update_water_level()

    def update_water_level(self):
        initial_water_level = self.water_level_slider.value()
        self.h0 = np.maximum(0, initial_water_level - self.z0)

    def update_composite_image(self, z, h):
        m = 10
        b = np.clip(h, 0, m) / m * 128

        land = np.zeros((z.shape[0], z.shape[1], 3))
        land[:, :, 1] = 128 - b

        water = np.zeros_like(land)
        water[:, :, 2] = b

        imgarr = (land + water).astype(np.uint8)
        self.img.setImage(imgarr)

    def update_water_image(self, z, h):
        m = 10
        b = (np.clip(h, 0, m) / m * 255).astype(np.uint8)
        self.img.setImage(b)

    def erode_terrain(self):
        dt = 0.1
        K_c = self.sediment_slider.value() / 10
        self.create_terrain()
        self.update_water_level()
        rainfall = 10 ** self.rainfall_slider.value()
        self.r0 = np.zeros_like(self.z0) + rainfall

        engine = FastErosionEngine(self.z0, self.h0, self.r0)
        for _ in tqdm(range(self.iterations_slider.value())):
            engine.update(dt, K_c)
        self.update_composite_image(engine.z, engine.h)
        del engine


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(1280, 720)
    w.show()
    sys.exit(app.exec_())
