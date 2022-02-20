import numpy as np
import scipy as sp
from scipy import linalg
from  datetime import datetime
import pickle
import random

from typing import Union, List, Tuple


def spare_matrix_Abt(m: int,n: int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,n), wektora b (n,)  i pomocniczego wektora t (m,) zawierających losowe wartości
    Parameters:
    m(int): ilość wierszy macierzy A
    n(int): ilość kolumn macierzy A
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,n) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m,int) and isinstance(n,int):
        t = np.transpose(np.linspace(0, 1, m))
        b = np.cos(4*t)
        A = np.vander(t,n)
        A = np.fliplr(A)
        return (A,b)
    return None


def square_from_rectan(A: np.ndarray, b: np.ndarray):
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników na kwadratowy układ równań. Funkcja ma zwrócić nową macierz współczynników  i nowy wektor współczynników
    Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (n,n) i wektorem (n,)
             Jeżeli dane wejściowe niepoprawne funkcja zwraca None
     """
    if isinstance(A,np.ndarray) and isinstance(b,np.ndarray):
        A_2 = np.transpose(A).dot(A)
        b_2 = np.transpose(A).dot(b)
        return A_2,b_2


    return None



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,n) zawierająca współczynniki równania
      x: wektor x (n,) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów
      """
    if all(isinstance(k,np.ndarray)  for k in [A,x,b]):
        residum = A@x - b
        return np.linalg.norm(residum)
    return None



t = np.random.uniform(0,1,(3,))

print(t)
A = np.vander(t,5)
print(A)
print('')
print(np.fliplr(A))
