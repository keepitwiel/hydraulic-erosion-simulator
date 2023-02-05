import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QWidget, QComboBox,
    QPlainTextEdit
)
import pyqtgraph as pg
import numpy as np
from tqdm import tqdm

from gui_utils import (
    Slider,
    initialize_engine,
    get_composite_image,
    get_terrain_height_image,
    get_surface_height_image,
    get_water_level_image,
    get_velocity_image,
    get_sediment_image,
    get_rainfall_image,
)


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
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
        self._add_slider("Random seed", 0, 255, 1, 0, lambda x: x)
        self._add_slider("Smoothness", 0, 10, 1, 6, lambda x: 0.1*x)
        self._add_slider("Map size", 7, 9, 1, 8, lambda x: 2**x)
        self._add_slider("Random amplitude", 0, 8, 4, 4, lambda x: 2**x)
        self._add_slider("Slope amplitude", 0, 8, 4, 4, lambda x: 2**x)
        self._add_slider("Sea level", -256, 256, 64, 0, lambda x: x)
        self._add_slider("Rainfall", -4, 2, 1, 1, lambda x: 10**x)
        self._add_slider("Sediment capacity constant", -5, 0, 1, -5, lambda x: 10**x)
        self._add_slider("Evaporation constant", -5, 0, 1, -5, lambda x: 10**x)
        self._add_slider("Number of iterations", 0, 3, 1, 2, lambda x: 10**x)
        self._add_slider("Time delta per step", -3, -1, 1, -1, lambda x: 10**x)

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
        self.mode_box.addItem("velocity")
        self.mode_box.addItem("sediment")
        self.mode_box.addItem("rainfall")
        self.mode_box.textActivated.connect(self.set_mode)
        self.layout_right.addWidget(self.mode_box)

        # image
        self.img = pg.RawImageWidget()
        self.layout_right.addWidget(self.img)

        # text
        font = QFont("Courier New")
        self.engine_textbox = QPlainTextEdit()
        self.engine_textbox.setMaximumHeight(120)
        self.engine_textbox.setFont(font)
        self.layout_right.addWidget(self.engine_textbox)

        self.simulation_textbox = QPlainTextEdit()
        self.simulation_textbox.setMaximumHeight(120)
        self.simulation_textbox.setFont(font)
        self.layout_right.addWidget(self.simulation_textbox)

        # final
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

        seed = self.sliders["Random seed"].mapped_value()
        map_size = self.sliders["Map size"].mapped_value()
        sea_level = self.sliders["Sea level"].mapped_value()
        random_amplitude = self.sliders["Random amplitude"].mapped_value()
        slope_amplitude = self.sliders["Slope amplitude"].mapped_value()
        smoothness = self.sliders["Smoothness"].mapped_value()
        rainfall = self.sliders["Rainfall"].mapped_value()

        self.engine = initialize_engine(seed, map_size, sea_level, random_amplitude, slope_amplitude, smoothness, rainfall)
        self.engine_textbox.setPlainText(f"""\
seed            : {seed: 3d}
map size        : {map_size: 3d}
sea level       : {sea_level: 3d}
random amplitude: {random_amplitude: 3d}
slope amplitude : {slope_amplitude: 3d}
smoothness      : {smoothness: 3.1f}
rainfall        : {rainfall: 3.1f}
"""
        )
        self.update_images()
        self.display_image()

    def update_images(self):
        if self.engine is not None:
            self.images["composite"] = get_composite_image(self.engine.z, self.engine.h, self.engine.u, self.engine.v)
            self.images["water level"] = get_water_level_image(self.engine.z, self.engine.h)
            self.images["terrain height"] = get_terrain_height_image(self.engine.z, self.engine.h)
            self.images["surface height"] = get_surface_height_image(self.engine.z, self.engine.h)
            self.images["velocity"] = get_velocity_image(self.engine.h, self.engine.u, self.engine.v)
            self.images["sediment"] = get_sediment_image(self.engine.s)
            self.images["rainfall"] = get_rainfall_image(self.engine.r)

    def erode_terrain(self):
        dt = self.sliders["Time delta per step"].mapped_value()
        k_c = self.sliders["Sediment capacity constant"].mapped_value()
        k_e = self.sliders["Evaporation constant"].mapped_value()
        n_iters = self.sliders["Number of iterations"].mapped_value()
        for _ in tqdm(range(n_iters)):
            self.engine.update(dt, k_c, k_e)

        self.simulation_textbox.setPlainText(f"""Sediment capacity constant: {k_c}
Evaporation constant      : {k_e}
Number of iterations      : {n_iters}

Total land volume         : {np.sum(self.engine.z):3.0f}
Total sediment volume     : {np.sum(self.engine.s):3.0f}
Total water volume        : {np.sum(self.engine.h):3.0f}
"""
        )
        self.update_images()
        self.display_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(1000, 750)
    w.show()
    sys.exit(app.exec_())
