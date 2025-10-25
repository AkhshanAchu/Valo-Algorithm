import numpy as np

class TestFunction:
    def __init__(self, func, name, x_min, f_min, lb=None, ub=None):
        self.func = func
        self.name = name
        self.x_min = np.array(x_min)
        self.f_min = f_min
        self.lb = np.array(lb) if lb is not None else None
        self.ub = np.array(ub) if ub is not None else None

    def __call__(self, x):
        return self.func(x)


# Define functions as before
def ackley(x):
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def griewank(x):
    x = np.asarray(x)
    sum1 = np.sum(x**2 / 4000)
    prod = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum1 - prod + 1

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def levi_n13(x):
    x1, x2 = x
    return np.sin(3 * np.pi * x1)**2 + (x1 - 1)**2 * (1 + np.sin(3 * np.pi * x2)**2) + (x2 - 1)**2 * (1 + np.sin(2 * np.pi * x2)**2)

def cross_in_tray(x):
    x1, x2 = x
    return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))) + 1)**0.1

def drop_wave(x):
    x1, x2 = x
    return - (1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2)

def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


# Create registry
FUNCTIONS = [
    TestFunction(ackley, 'ackley', [0, 0], 0, lb=[-5.0, -5.0], ub=[5.0, 5.0]),
    TestFunction(rastrigin, 'rastrigin', [0, 0], 0, lb=[-5.12, -5.12], ub=[5.12, 5.12]),
    TestFunction(griewank, 'griewank', [0, 0], 0, lb=[-600, -600], ub=[600, 600]),
    TestFunction(rosenbrock, 'rosenbrock', [1, 1], 0, lb=[-5, -5], ub=[10, 10]),
    TestFunction(levi_n13, 'levi_n13', [1, 1], 0, lb=[-10, -10], ub=[10, 10]),
    TestFunction(cross_in_tray, 'cross_in_tray', [1.34941, -1.34941], -2.06261, lb=[-10, -10], ub=[10, 10]),
    TestFunction(eggholder, 'eggholder', [512, 404.2319], -959.6407, lb=[-512, -512], ub=[512, 512]),
    TestFunction(drop_wave, 'drop_wave', [0, 0], -1, lb=[-5.12, -5.12], ub=[5.12, 5.12]),
]
    