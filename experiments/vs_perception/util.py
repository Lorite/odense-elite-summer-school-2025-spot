import time
import numpy as np


class RunningMedian():

    def __init__(self, num_values):
        self.values = [0 for _ in range(num_values)]
        self.idx = 0

    def add_value(self, value):
        self.values[self.idx] = value
        self.idx += 1
        self.idx %= len(self.values)

    def get(self):
        return np.median(self.values)


class RunningMedianTuple():

    def __init__(self, num_values, tuple_size):
        self.running_medians = [RunningMedian(
            num_values) for _ in range(tuple_size)]

    def add_value(self, value):
        for i in range(len(self.running_medians)):
            self.running_medians[i].add_value(value[i])

    def get(self):
        medians = [running_median.get()
                   for running_median in self.running_medians]
        return tuple(medians)


class FPSMeasurement():
    def __init__(self):
        self.t = time.time()
        self.fps = None

    def frame(self):
        t_new = time.time()
        t_diff = t_new - self.t
        self.fps = 1.0 / t_diff
        self.t = t_new

    def get_fps(self):
        return self.fps
