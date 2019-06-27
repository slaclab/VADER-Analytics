__author__ = 'Jiafan Yu'


class SimpleAverage:
    def __init__(self, step_ahead, window_size=168):
        self.step_ahead = step_ahead
        self.window_size = window_size
        pass

    def predict(self, X):
        x_load = X[:, range(self.window_size, 2 * self.window_size)]
        recent_load_indices = range(self.step_ahead, self.window_size, 24)
        return x_load[:, recent_load_indices].mean(axis=1)


