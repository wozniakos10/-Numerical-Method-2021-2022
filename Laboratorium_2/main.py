import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import string
import random


def compare_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                 xlabel: str, ylabel: str, title: str, label1: str, label2: str):
    if x1.shape != y1.shape or x2.shape != y2.shape or min(x1.shape) == 0 or min(x2.shape) == 0:
        return None
    fig, ax = plt.subplots()
    ax.plot(x1, y1, "b", linewidth=4, label=label1)
    ax.plot(x2, y2, "r", linewidth=2, label=label2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.legend()

    return fig


def parallel_plot(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
                  x1label: str, y1label: str, x2label: str, y2label: str, title: str, orientation: str):
    if x1.shape != y1.shape or x2.shape != y2.shape or np.min(x1.shape) == 0 or np.min(x2.shape) == 0:
        return None
    if orientation == '-':
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(title)
        ax1.set(xlabel=x1label, ylabel=y1label)
        ax2.set(xlabel=x2label, ylabel=y2label)
        ax1.plot(x1, y1, 'm', linewidth=3)
        ax1.grid()
        ax2.plot(x2, y2, 'c', linewidth=3)
        ax2.grid()
        return fig
    elif orientation == '|':
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(title)
        ax1.set(xlabel=x1label, ylabel=y1label)
        ax2.set(xlabel=x2label, ylabel=y2label)
        ax1.plot(x1, y1, 'm', linewidth=3)
        ax1.grid()
        ax2.plot(x2, y2, 'c', linewidth=3)
        ax2.grid()
        return fig
    else:
        return None


def log_plot(x: np.ndarray, y: np.ndarray, xlabel: np.str, ylabel: str, title: str, log_axis: str):
    if x.shape != y.shape or min(x.shape) == 0:
        return None
    if log_axis == 'y':
        fig, ax = plt.subplots()
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.plot(x, y, 'k')
        ax.set_yscale("log")
        ax.grid()
        return fig
    elif log_axis == 'x':
        fig, ax = plt.subplots()
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.plot(x, y, 'r')
        ax.grid()
        ax.set_xscale("log")
        return fig
    elif log_axis == 'xy':
        fig, ax = plt.subplots()
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.plot(x, y, 'g')
        ax.grid()
        ax.set_yscale("log")
        ax.set_xscale("log")
        return fig
    else:
        return None
