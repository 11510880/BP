class EdgeMessage:
    """
    Three-Gaussian model
    """
    def __init__(self):
        self.means = [0] * 3
        self.vars = [0] * 3
        self.coef = [0] * 3

    def __iter__(self, means, vars, coef):
        self.means = means
        self.coef = coef
        self.vars = vars