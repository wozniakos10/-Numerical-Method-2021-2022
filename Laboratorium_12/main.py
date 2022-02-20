import numpy as np
from typing import Union, Callable
import matplotlib.pyplot as plt

def solve_euler(fun: Callable, t_span: np.array, y0: np.array):
    '''
    Funkcja umożliwiająca rozwiązanie układu równań różniczkowych z wykorzystaniem metody Eulera w przód.

    Parameters:
    fun: Prawa strona równania. Podana funkcja musi mieć postać fun(t, y).
    Tutaj t jest skalarem i istnieją dwie opcje dla ndarray y: Może mieć kształt (n,); wtedy fun musi zwrócić array_like z kształtem (n,).
    Alternatywnie może mieć kształt (n, k); wtedy fun musi zwrócić tablicę typu array_like z kształtem (n, k), tj. każda kolumna odpowiada jednej kolumnie w y.
    t_span: wektor czasu dla którego ma zostać rozwiązane równanie
    y0: warunke początkowy równanai o wymiarze (n,)
    Results:
    (np.array): macierz o wymiarze (n,m) zawierająca w wkolumnach kolejne rozwiązania fun w czasie t_span.

    '''

    if isinstance(fun,Callable) and isinstance(t_span,np.ndarray) and isinstance(y0,np.ndarray):
        y = np.zeros((t_span.size, y0.size))

        y[0] = y0
        for i in range(1,t_span.size):
            h = t_span[i] - t_span[i-1]
            y[i] = y[i-1] + h * fun(t_span[i-1], y[i-1])

        return np.array(y)
    else:
        raise ValueError('Błędne dane wejściowe')

