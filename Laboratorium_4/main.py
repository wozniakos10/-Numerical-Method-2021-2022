##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import barycentric_interpolate


from typing import Union, List, Tuple

def f_c_n(x : Union[int,float,np.ndarray]):
    if isinstance(x,(int,float,np.ndarray)):
        return np.sign(x)*x + x**2



def f_r_1(x: Union[int,float,np.ndarray]):
    if isinstance(x,(int,float,np.ndarray)):
        return np.sign(x) * x**2

    return None

def f_r_3(x: Union[int,float,np.ndarray]):
    if isinstance(x,(int,float,np.ndarray)):
        return (np.abs(np.sin(5*x)))**3
    return None

def f_an_1(x: Union[int,float,np.ndarray]):
    if isinstance(x,(int,float,np.ndarray)):
        return 1/(1+x**2)
    return None


def f_an_2(x: Union[int, float, np.ndarray]):
    if isinstance(x, (int, float, np.ndarray)):
        return  1 / (1 + 25 * x ** 2)
    return None


def f_an_3(x: Union[int, float, np.ndarray]):
    if isinstance(x, (int, float, np.ndarray)):
        return  1 / (1 + 100 * x ** 2)
    return None



def f_sgn(x : Union[int,float,np.ndarray]):
    if isinstance(x,(int,float,np.ndarray,List)):
        return np.sign(x)
    return None

def chebyshev_nodes(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor zawierający węzły czybyszewa w postaci wektora (n+1,)
    
    Parameters:
    n(int): numer ostaniego węzła Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n,int):
        nodes = np.zeros(n+1)
        for k in range(n+1):
            nodes[k] = np.cos((k*np.pi)/n)
        return np.array(nodes)
    return None


    
def bar_czeb_weights(n:int=10)-> np.ndarray:
    """Funkcja tworząca wektor wag dla węzłów czybyszewa w postaci (n+1,)
    
    Parameters:
    n(int): numer ostaniej wagi dla węzłów Czebyszewa. Wartość musi być większa od 0.
     
    Results:
    np.ndarray: wektor wag dla węzłów Czybyszewa o rozmiarze (n+1,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(n,int):
        czeb_wei = np.zeros(n+1)
        for j in range(n+1):
            if j == 0 or j == n:
                omega = 0.5
            if j >0 and j < n:
                omega = 1
            czeb_wei[j] = ((-1)**j) * omega
        return czeb_wei

    return None


    
def  barycentric_inte(xi:np.ndarray,yi:np.ndarray,wi:np.ndarray,x:np.ndarray)-> np.ndarray:
    """Funkcja przprowadza interpolację metodą barycentryczną dla zadanych węzłów xi
        i wartości funkcji interpolowanej yi używając wag wi. Zwraca wyliczone wartości
        funkcji interpolującej dla argumentów x w postaci wektora (n,) gdzie n to dłógość
        wektora n. 
    
    Parameters:
    xi(np.ndarray): węzły interpolacji w postaci wektora (m,), gdzie m > 0
    yi(np.ndarray): wartości funkcji interpolowanej w węzłach w postaci wektora (m,), gdzie m>0
    wi(np.ndarray): wagi interpolacji w postaci wektora (m,), gdzie m>0
    x(np.ndarray): argumenty dla funkcji interpolującej (n,), gdzie n>0 
     
    Results:
    np.ndarray: wektor wartości funkcji interpolujący o rozmiarze (n,). 
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(xi,np.ndarray) and isinstance(yi,np.ndarray) and isinstance(wi,np.ndarray) and isinstance(x,np.ndarray):

        Y = []
        try:
            for x in np.nditer(x):
                if x in xi:
                    # omijamy dzielenie przez 0
                    Y.append(yi[np.where(xi == x)[0][0]])
                else:
                    # wzór w drugiej formie
                    L = wi / (x - xi)
                    Y.append(yi @ L / sum(L))
            return np.array(Y)
        except:
            return None





    return None

def L_inf(xr:Union[int, float, List, np.ndarray],x:Union[int, float, List, np.ndarray])-> float:
    """Obliczenie normy  L nieskończonośćg. 
    Funkcja powinna działać zarówno na wartościach skalarnych, listach jak i wektorach biblioteki numpy.
    
    Parameters:
    xr (Union[int, float, List, np.ndarray]): wartość dokładna w postaci wektora (n,)
    x (Union[int, float, List, np.ndarray]): wartość przybliżona w postaci wektora (n,1)
    
    Returns:
    float: wartość normy L nieskończoność,
                                    NaN w przypadku błędnych danych wejściowych
    """

    if all(isinstance(i, (int, float)) for i in [xr, x]):
        return np.abs(xr - x)

    elif all(isinstance(i, np.ndarray) for i in [xr, x]):
        if xr.shape == x.shape:
            return max(np.abs(xr - x))
        else:
            return np.NaN

    elif all(isinstance(i, list) for i in [xr, x]):
        return np.abs(max(xr) - max(x))

    else:
        return np.NaN