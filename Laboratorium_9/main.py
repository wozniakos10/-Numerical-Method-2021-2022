import numpy as np
import scipy
import pickle
import typing
import math
import types
import pickle 
from inspect import isfunction


from typing import Union, List, Tuple

def fun(x):
    return np.exp(-2*x)+x**2-1

def dfun(x):
    return -2*np.exp(-2*x) + 2*x

def ddfun(x):
    return 4*np.exp(-2*x) + 2


def bisection(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą bisekcji.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(epsilon, float) or \
            not isinstance(iteration, int) or not isfunction(f) or f(a) * f(b) >= 0 or iteration <= 0:
        return None

    else:

        for iter in range(iteration):
            c = (a + b) / 2
            if np.abs(f(c)) < epsilon:
                return c, iter
            else:
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
        return c, iter


def secant(a: Union[int,float], b: Union[int,float], f: typing.Callable[[float], float], epsilon: float, iteration: int) -> Tuple[float, int]:
    '''funkcja aproksymująca rozwiązanie równania f(x) = 0 na przedziale [a,b] metodą siecznych.

    Parametry:
    a - początek przedziału
    b - koniec przedziału
    f - funkcja dla której jest poszukiwane rozwiązanie
    epsilon - tolerancja zera maszynowego (warunek stopu)
    iteration - ilość iteracji

    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and isinstance(epsilon,float) and isinstance(iteration,int) and callable(f):
        if f(a) * f(b) < 0:
            for iter in range(iteration):

                x = (f(b) * a - f(a) * b)/(f(b) - f(a))
                if f(x) * f(a) > 0:
                    a = x
                    b = b
                if f(x)*f(b) >0:
                    a = a
                    b = x

                if abs(f(x)) < epsilon or abs(b - a) < epsilon:
                    return x,iter

            return (f(b) * a - f(a) * b)/(f(b) - f(a)), iteration




    return None

def newton(f: typing.Callable[[float], float], df: typing.Callable[[float], float], ddf: typing.Callable[[float], float], a: Union[int,float], b: Union[int,float], epsilon: float, iteration: int) -> Tuple[float, int]:
    ''' Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.
    Parametry: 
    f - funkcja dla której jest poszukiwane rozwiązanie
    df - pochodna funkcji dla której jest poszukiwane rozwiązanie
    ddf - druga pochodna funkcji dla której jest poszukiwane rozwiązanie
    a - początek przedziału
    b - koniec przedziału
    epsilon - tolerancja zera maszynowego (warunek stopu)
    Return:
    float: aproksymowane rozwiązanie
    int: ilość iteracji
    '''

    if  isfunction(f) and  isfunction(df) and isfunction(ddf) and isinstance(a, (int, float))  \
            and isinstance(b, (int, float)) and isinstance(epsilon, float) and isinstance(iteration, int) \
            and f(a) * f(b) < 0 and iteration > 0 and df(a) * df(b) >= 0 and ddf(a) * ddf(b) >= 0:

            for iter in range(1, iteration + 1):
                c = b - f(b) / df(b)

                if np.abs(f(c)) < epsilon:
                    return c, iter
                else:
                    b = c

            return c, iter


    else:
        return None

