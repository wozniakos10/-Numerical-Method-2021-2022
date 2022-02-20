import numpy as np
import scipy as sp
import pickle

from typing import Union, List, Tuple, Optional


def diag_dominant_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Macierz A ma być diagonalnie zdominowana, tzn. wyrazy na przekątnej sa wieksze od pozostałych w danej kolumnie i wierszu
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: macierz diagonalnie zdominowana o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m,int) and m > 0 :
        A = np.random.randint(0,10,(m,m))
        b = np.random.randint(0,10,(m,1))
        m_diag_sum = np.sum(A,axis =0) + np.sum(A,axis = 1)
        m_diag_sum = np.diag(m_diag_sum)
        A = A + m_diag_sum
        return A,b


    return None


def is_diag_dominant(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest diagonalnie zdominowana
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A,np.ndarray):
        if len(A.shape) == 2:
            if np.shape(A)[0] == np.shape(A)[1]:

                return all((2 * np.abs(np.diag(A))) >= sum(np.abs(A), 0) + sum(np.abs(A),1) - 2*np.abs(np.diag(A)))

    return None


def symmetric_matrix_A_b(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych z przedziału 0, 9
    Parameters:
    m int: wymiary macierzy i wektora
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: symetryczną macierz o rozmiarze (m,m) i wektorem (m,)
                                   Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """

    if isinstance(m,int) and m >0:
        A = np.random.randint(0,10,(m,m))
        b = np.random.randint(0,10,(m,))
        A_sym = A + A.T
        return A_sym,b

    return None


def is_symmetric(A: np.ndarray) -> bool:
    """Funkcja sprawdzająca czy macierzy A (m,m) jest symetryczna
    Parameters:
    A np.ndarray: macierz wejściowa
    
    Returns:
    bool: sprawdzenie warunku 
          Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(A, np.ndarray):
        if len(A.shape) == 2:
            if A.shape[0] == A.shape[1]:
                # True jeżeli odpowiadające elementy różnią się od siebie o max 1e-05
                if np.allclose(A, A.T, 1e-05, 1e-05):
                    return True
                return False

    return None


def solve_jacobi(A: np.ndarray, b: np.ndarray, x_init: np.ndarray,
                 epsilon: Optional[float] = 1e-8, maxiter: Optional[int] = 100) -> Tuple[np.ndarray, int]:
    """Funkcja tworząca zestaw składający się z macierzy A (m,m), wektora b (m,) o losowych wartościach całkowitych
    Parameters:
    A np.ndarray: macierz współczynników
    b np.ndarray: wektor wartości prawej strony układu
    x_init np.ndarray: rozwiązanie początkowe
    epsilon Optional[float]: zadana dokładność
    maxiter Optional[int]: ograniczenie iteracji
    
    Returns:
    np.ndarray: przybliżone rozwiązanie (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    int: iteracja
    """
    if not isinstance(maxiter,int) or maxiter < 0:
        return None
    else:
        try:
            D = np.diag(np.diag(A))
            LU = A - D
            x = x_init
            D_inv = np.diag(1 / np.diag(D))

            for i in range(maxiter):
                x_new = np.dot(D_inv, b - np.dot(LU, x))
                r_norm = np.linalg.norm(x_new - x)

                if r_norm < epsilon:
                    return x_new, i
                x = x_new
            return x, maxiter
        except:
            return None


def random_matrix_Ab(m:int):
    """Funkcja tworząca zestaw składający się z macierzy A (m,m) i wektora b (m,)  zawierających losowe wartości
    Parameters:
    m(int): rozmiar macierzy
    Results:
    (np.ndarray, np.ndarray): macierz o rozmiarze (m,m) i wektorem (m,)
                Jeżeli dane wejściowe niepoprawne funkcja zwraca None
    """
    if isinstance(m,int) and m >= 1:
        b = np.random.randint(0,101,size = m)
        A = np.random.randint(0,101,size = (m,m))
        return A,b

 # Funckja do wyliczania normy
def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

      Parameters:
      A: macierz A (m,m) zawierająca współczynniki równania
      x: wektor x (m.) zawierający rozwiązania równania
      b: wektor b (m,) zawierający współczynniki po prawej stronie równania

      Results:
      (float)- wartość normy residuom dla podanych parametrów"""

    if len(A[0]) == len(x) == len(b):
        r = b - A @ x
        return np.linalg.norm(r)

    return None



