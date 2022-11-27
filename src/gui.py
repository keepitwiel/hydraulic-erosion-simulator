import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton, QWidget,
)
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from height_map import generate_height_map
from fast_erosion_engine import FastErosionEngine


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
        self.z0 = None
        self.h0 = None
        self.r0 = None
        self.seed = 42

        layout_main = QHBoxLayout()
        self.setLayout(layout_main)

        # left column
        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()

        # erode_button
        self.erode_button = QPushButton(text="Erode")
        self.erode_button.setMinimumHeight(100)
        self.erode_button.pressed.connect(self.erode_terrain)
        layout_left.addWidget(self.erode_button)

        # map size slider
        map_size_label = QLabel("Map size")
        map_size_label.setMaximumHeight(20)
        layout_left.addWidget(map_size_label)
        self.map_size_slider = QSlider(Qt.Horizontal)
        self.map_size_slider.setMaximumHeight(20)
        self.map_size_slider.setMinimum(5)
        self.map_size_slider.setMaximum(9)
        self.map_size_slider.setTickInterval(1)
        self.map_size_slider.setSingleStep(1)
        self.map_size_slider.setValue(7)
        self.map_size_slider.valueChanged.connect(self.create_terrain)
        layout_left.addWidget(self.map_size_slider)

        # water level slider
        water_level_label = QLabel("Water level")
        water_level_label.setMaximumHeight(20)
        layout_left.addWidget(water_level_label)
        self.water_level_slider = QSlider(Qt.Horizontal)
        self.water_level_slider.setMaximumHeight(20)
        self.water_level_slider.setMinimum(-50)
        self.water_level_slider.setMaximum(50)
        self.water_level_slider.setValue(0)
        self.water_level_slider.valueChanged.connect(self.update_water_level)
        layout_left.addWidget(self.water_level_slider)

        # rainfall slider
        rainfall_label = QLabel("Rainfall")
        rainfall_label.setMaximumHeight(20)
        layout_left.addWidget(rainfall_label)
        self.rainfall_slider = QSlider(Qt.Horizontal)
        self.rainfall_slider.setMaximumHeight(20)
        self.rainfall_slider.setMinimum(-6)
        self.rainfall_slider.setMaximum(0)
        self.rainfall_slider.setTickInterval(1)
        self.rainfall_slider.setSingleStep(1)
        self.rainfall_slider.setValue(-6)
        layout_left.addWidget(self.rainfall_slider)

        # sediment capacity slider
        sediment_label = QLabel("Sediment capacity constant")
        sediment_label.setMaximumHeight(20)
        layout_left.addWidget(sediment_label)
        self.sediment_slider = QSlider(Qt.Horizontal)
        self.sediment_slider.setMaximumHeight(20)
        self.sediment_slider.setMinimum(0)
        self.sediment_slider.setMaximum(10)
        self.sediment_slider.setTickInterval(1)
        self.sediment_slider.setSingleStep(1)
        self.sediment_slider.setValue(1)
        layout_left.addWidget(self.sediment_slider)

        # iterations slider
        iterations_label = QLabel("Number of iterations")
        iterations_label.setMaximumHeight(20)
        layout_left.addWidget(iterations_label)
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.iterations_slider.setMaximumHeight(20)
        self.iterations_slider.setMinimum(0)
        self.iterations_slider.setMaximum(1000)
        self.iterations_slider.setTickInterval(100)
        self.iterations_slider.setSingleStep(100)
        self.iterations_slider.setValue(100)
        layout_left.addWidget(self.iterations_slider)

        layout_main.addLayout(layout_left, stretch=1)

        # right column
        self.img = pg.RawImageWidget()
        arr = np.zeros(shape=(128, 128), dtype=np.uint8)
        self.img.setImage(arr)
        layout_right.addWidget(self.img)

        layout_main.addLayout(layout_right, stretch=3)

        # final stuff
        self.create_terrain()
        self.update_water_level()

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
