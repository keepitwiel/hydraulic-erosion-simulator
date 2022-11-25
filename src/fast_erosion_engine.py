# this implementation is based on the paper
# https://hal.inria.fr/inria-00402079/document
#
# original implementation:
# https://github.com/Huw-man/Interactive-Erosion-Simulator-on-GPU

# "alternative" implementation:
# https://github.com/karhu/terrain-erosion/blob/master/Simulation/FluidSimulation.cpp

import numpy as np
from src.fast_erosion_algorithm import update


class FastErosionEngine:
    def __init__(self, z0, h0, r0):
        self.z = z0
        self.h = h0
        self.r = r0

        self.s = np.zeros_like(z0)

        self.fL = np.zeros_like(z0)
        self.fR = np.zeros_like(z0)
        self.fT = np.zeros_like(z0)
        self.fB = np.zeros_like(z0)

        self.u = np.zeros_like(z0)
        self.v = np.zeros_like(z0)

    def update(self, dt):
        (
            self.z, self.h, self.s,
            self.fL, self.fR, self.fT, self.fB,
            self.u, self.v
        ) = update(
            self.z, self.h, self.r, self.s,
            self.fL, self.fR, self.fT, self.fB,
            dt,
        )
