import numpy as np
import matplotlib.pyplot as plt


def create_beta_zero(t):
    beta_0 = np.zeros_like(t)
    beta_0[-0.5 < t < 0.5] = 1.
    beta_0[t == 0.5] = 0.5

    return beta_0

def create_n_order_spline(t, n):
    assert 0 <= n <= 3, "n must be 0 <= n <= 3"
    beta_0 = create_beta_zero(t)
    if n == 0:
        return beta_0

    beta_n = np.copy(beta_0)
    for i in range(1, n + 1):
        beta_n = np.convolve(beta_n, beta_0, mode='same')
    
    return beta_n

def SplineExpansion(d, t, n):
    beta_n = create_n_order_spline(t, n)
    return np.convolve(d, beta_n, mode='same')
