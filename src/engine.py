import numpy as np
from algorithm import _update


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
        self.g = np.zeros_like(z0)

    def update(self, dt, k_c, k_e, erosion_flag=True):
        (
            self.z, self.h, self.s,
            self.fL, self.fR, self.fT, self.fB,
            self.u, self.v, self.g
        ) = _update(
            self.z, self.h, self.r, self.s,
            self.fL, self.fR, self.fT, self.fB,
            dt, k_c=k_c, k_e=k_e,
            erosion_flag=erosion_flag,
        )
