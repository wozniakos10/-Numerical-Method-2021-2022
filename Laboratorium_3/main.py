import numpy as np
import scipy
import pickle

from typing import Union, List, Tuple


def absolut_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu bezwzględnego. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu bezwzględnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    if not isinstance(v, (int, float, list, np.ndarray)) or not isinstance(v_aprox, (int, float, list, np.ndarray)):
        return np.nan
    elif isinstance(v, (list, np.ndarray)) and isinstance(v_aprox, (list, np.ndarray)) and len(v) != len(v_aprox):
        return np.nan

    value = abs(np.array(v) - np.array(v_aprox))

    return value


def relative_error(v: Union[int, float, List, np.ndarray], v_aprox: Union[int, float, List, np.ndarray]) -> Union[int, float, np.ndarray]:
    """Obliczenie błędu względnego.
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach/macierzach biblioteki numpy.
    
    Parameters:
    v (Union[int, float, List, np.ndarray]): wartość dokładna 
    v_aprox (Union[int, float, List, np.ndarray]): wartość przybliżona
    
    Returns:
    err Union[int, float, np.ndarray]: wartość błędu względnego,
                                       NaN w przypadku błędnych danych wejściowych
    """
    abs = absolut_error(v, v_aprox)
    if abs is np.NaN:
        return np.NaN

    elif isinstance(v, np.ndarray):
        return np.divide(abs, v)

    elif isinstance(v, (int, float)) and v == 0:
        return np.NaN

    elif isinstance(v, np.ndarray) and not v.any():
        return np.NaN

    elif isinstance(abs, np.ndarray) and isinstance(v, list):
        output_list = np.zeros(len(v))

        for i in range(len(v)):
            if v[i] == 0:
                return np.NaN

            output_list[i] = abs[i] / v[i]

        return output_list

    else:
        return abs / v


def p_diff(n: int, c: float) -> float:
    """Funkcja wylicza wartości wyrażeń P1 i P2 w zależności od n i c.
    Następnie zwraca wartość bezwzględną z ich różnicy.
    Szczegóły w Zadaniu 2.
    
    Parameters:
    n Union[int]: 
    c Union[int, float]: 
    
    Returns:
    diff float: różnica P1-P2
                NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n,int) and isinstance(c,(float,int)):
        b = 2**n
        P1 = b - b + c
        P2 = b + c - b
        diff = float(abs(P1 - P2))
        return diff

    return np.NaN


def exponential(x: Union[int, float], n: int) -> float:
    """Funkcja znajdująca przybliżenie funkcji exp(x).
    Do obliczania silni można użyć funkcji scipy.math.factorial(x)
    Szczegóły w Zadaniu 3.
    
    Parameters:
    x Union[int, float]: wykładnik funkcji ekspotencjalnej 
    n Union[int]: liczba wyrazów w ciągu
    
    Returns:
    exp_aprox float: aproksymowana wartość funkcji,
                     NaN w przypadku błędnych danych wejściowych
    """

    if isinstance(x,(int,float)) and isinstance(n,int):
        if n < 0 :
            return np.NaN
        exp_aprox = 0
        for i in range(n):
            exp_aprox += (1 / np.math.factorial(i)) * x **i
        if exp_aprox is int:
            float(exp_aprox)
        return exp_aprox

    else:
        return np.NaN


def coskx1(k: int, x: Union[int, float]) -> float:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 1.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx float: aproksymowana wartość funkcji,
                 NaN w przypadku błędnych danych wejściowych
    """


    if isinstance(k,int) and isinstance(x,(int,float)):
        if k < 0:
            return np.NaN
        if k == 0:
            return float(1)
        elif k == 1:
            coskx = np.cos(k*x)
            return float(coskx)
        else:
            return  float(2*np.cos(x) * coskx1(k-1,x) - coskx1(k-2,x))

    else:
        return np.NaN


def coskx2(k: int, x: Union[int, float]) -> Tuple[float, float]:
    """Funkcja znajdująca przybliżenie funkcji cos(kx). Metoda 2.
    Szczegóły w Zadaniu 4.
    
    Parameters:
    x Union[int, float]:  
    k Union[int]: 
    
    Returns:
    coskx, sinkx float: aproksymowana wartość funkcji,
                        NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(k,int) and isinstance(x,(int,float)):
        if k < 0:
            return np.NaN
        if k == 0 or x == 0 :
            return 1,0
        elif k == 1:
            coskx = np.cos(x)
            sinx = np.sin(x)
            return coskx,sinx
        else:
            return np.cos(x) * coskx2(k-1,x)[0] - np.sin(x)*coskx2(k-1,x)[1], np.sin(x) * coskx2(k-1,x)[0] + np.cos(x) * coskx2(k-1,x)[1]

    else:
        return np.NaN

def pi(n: int) -> float:
    """Funkcja znajdująca przybliżenie wartości stałej pi.
    Szczegóły w Zadaniu 5.
    
    Parameters:
    n Union[int, List[int], np.ndarray[int]]: liczba wyrazów w ciągu
    
    Returns:
    pi_aprox float: przybliżenie stałej pi,
                    NaN w przypadku błędnych danych wejściowych
    """
    if isinstance(n,int):
        if n <= 0 :
            return np.NaN
        pi_sum = 0
        for i in range(1,n+1):
            pi_sum += 1/(i**2)
        pi_aprox = np.sqrt(6*pi_sum)
        return pi_aprox

    else:
        return np.NaN




