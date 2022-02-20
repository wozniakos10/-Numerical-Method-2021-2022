import math
import numpy as np
from numpy.linalg import inv


def cylinder_area(r: float, h: float):
    if r <= 0 or h <= 0:
        return np.NaN
    else:
        return 2 * math.pi * (r ** 2) + 2 * math.pi * h*r


def fib(n: int):
    if n < 0:
        return np.NaN
    else:
        x = np.array([1,1])
        for i in range(1, n - 1):
            x = np.append(x, x[-1] + x[-2])
        return x


def matrix_calculations(a: float):
    x = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    return inv(x), np.transpose(x), np.linalg.det(x)


def custom_matrix(m: int, n: int):
    if m <= 0 or n <= 0:
        return None
    else:
        x = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if i > j:
                    x[i, j] = i
                else:
                    x[i, j] = j
        return x


