from shaker2d import update


class ShakerEngine:
    def __init__(self, z, h, k=0.99):
        self.z = z
        self.h = h
        self.k = k

    def update(self, r):
        self.h += r
        self.z, self.h = update(self.z, self.h, self.k)
