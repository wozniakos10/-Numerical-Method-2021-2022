import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp
from numpy.core._multiarray_umath import ndarray
from numpy.polynomial import polynomial as P
import pickle
import random
from numpy import linalg as LA


# zad1
def polly_A(x: np.ndarray):
    """Funkcja wyznaczajaca współczynniki wielomianu przy znanym wektorze pierwiastków.
    Parameters:
    x: wektor pierwiastków
    Results:
    (np.ndarray): wektor współczynników
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if isinstance(x,np.ndarray):
        return P.polyfromroots(x)
    
    return None

def roots_20(a: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray): wektor współczynników i miejsc zerowych w danej pętli
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(a,np.ndarray):

        a = np.array(a, dtype=float)
        random_value = 1e-10 * np.random.sample(len(a)) * np.random.randint(0,2,(len(a),))
        a += random_value

        return a, P.polyroots(a)



    return None


# zad 2

def frob_a(wsp: np.ndarray):
    """Funkcja zaburzająca lekko współczynniki wielomianu na postawie wyznaczonych współczynników wielomianu
        oraz zwracająca dla danych współczynników, miejsca zerowe wielomianu funkcją polyroots.
    Parameters:
    a: wektor współczynników
    Results:
    (np.ndarray, np. ndarray, np.ndarray, np. ndarray,): macierz Frobenusa o rozmiarze nxn, gdzie n-1 stopień wielomianu,
    wektor własności własnych, wektor wartości z rozkładu schura, wektor miejsc zerowych otrzymanych za pomocą funkcji polyroots

                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if isinstance(wsp,np.ndarray):

        frob = np.zeros((len(wsp) ,len(wsp) ))
        for i in range(len(frob)-1):

            frob[i][i +1] = 1
        for i in range(len(frob[-1])):
            frob[-1][i] = -wsp[i]




        return frob, LA.eigvals(frob),sp.schur(frob),P.polyroots(wsp)
    return None


