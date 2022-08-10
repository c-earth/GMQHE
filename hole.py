import numpy as np

class Hole():
    '''
    '''
    def __init__(self):
        pass

    def check(self, site):
        pass

    def __call__(self, site):
        return self.check(site)

class SquareHole2D(Hole):
    '''
    '''
    def __init__(self, a, center):
        super().__init__()
        self.a = a
        self.center = np.array(center)

    def check(self, site):
        pos = np.array(site.pos)
        return np.prod((np.absolute(pos - self.center) <= self.a/2)[:2])