import numpy as np


def linear_trend(x: np.array, coeff: float = 1, intercept: float = 0):
    return coeff * x + intercept


def exponential_trend(x: np.array, coeff: float = 1, intercept: float = 0):
    return np.exp(coeff * x) + intercept - 1


def seasonality(x: np.array, freq: float = 1, phi: float = 0, coeff: float = 1):
    return coeff * np.sin(2 * np.pi * freq * x + phi)


def autocorrelation_function(x):
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x - np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[: x.size // 2] / np.sum(xp**2)
