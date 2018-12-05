import numpy as np


class PolynomialModel(object):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3 + ...
    where the highest power of x is the order of the polynomial.
    """
    def __init__(self, order=1):
        self.order = int(order)
        self.n_params = self.order + 1
        names = {
            1: "Linear",
            2: "Quadratic",
            3: "Cubic",
            4: "Quartic",
            5: "Quintic",
            6: "Sextic",
            7: "Septic",
            8: "Octic",
        }
        if names.get(self.order):
            self.name = names.get(self.order) + " model"
        else:
            self.name = str(self.order) + "-order polynomial model"
        self.n_iter_default = self.order * 10

    def initial_guess(self, x, y):
        params = [np.random.normal(loc=np.mean(y), scale=np.std(y))]
        for i_term in range(self.order):
            params.append(np.random.normal(
                loc=0, scale=np.std(y) / np.std(x)**(i_term + 1)))
        return np.array(params)

    def evaluate(self, p, x):
        assert len(p) == self.n_params

        y = p[0]
        for i_term in range(self.order):
            y += p[i_term + 1] * x ** (i_term + 1)
        return y

    def to_points(self, p, x):
        x_min = np.min(x) - (np.max(x) - np.min(x)) / 8
        x_max = np.max(x) + (np.max(x) - np.min(x)) / 8
        x_c = np.linspace(x_min, x_max, num=300)
        y_c = self.evaluate(p, x_c)
        return x_c, y_c


all_models = [
    PolynomialModel(order=1),
    PolynomialModel(order=2),
    PolynomialModel(order=3),
    PolynomialModel(order=4),
    PolynomialModel(order=5),
    PolynomialModel(order=6),
]
