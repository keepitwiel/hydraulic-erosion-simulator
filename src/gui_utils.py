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
