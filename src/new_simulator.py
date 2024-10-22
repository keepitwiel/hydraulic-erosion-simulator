import numpy as np

from algo2 import update

class NewSimulator:
    def __init__(self, z0, h0, r0):
        self.z = z0
        self.h = h0
        self.r = r0
        self.v = np.zeros_like(z0)
        self.a_p = np.zeros_like(z0)
        self.a_v = np.zeros_like(z0)
        self.k = 5.0
        self.nu = 1.0
        self.dt = 0.1

    def update(self, dt, k_c, k_e, erosion_flag=True):
        self.h += self.r * dt
        update(self.z, self.h, self.v, self.a_p, self.a_v, self.k, self.nu, self.dt)
        self.h = np.maximum(0.0, self.h - 0.01 * dt)