import numpy as np
import matplotlib.pyplot as plt


def create_beta_zero(t):
    beta_0 = np.zeros_like(t)
    beta_0[np.bitwise_and(-0.5 < t, t < 0.5)] = 1.
    beta_0[t == 0.5] = 0.5
    return beta_0

def create_n_order_spline(t, n):
    assert 0 <= n <= 3, "n must be 0 <= n <= 3"
    beta_0 = create_beta_zero(t)
    if n == 0:
        return beta_0

    dt = t[1] - t[0]
    beta_n = np.copy(beta_0)
    for i in range(1, n + 1):
        beta_n = dt * np.convolve(beta_n, beta_0, mode='same')
        plt.plot(t, beta_n)
        plt.show()
    
    return beta_n

def resample_time(t, resample_rate=10):
    min_t = t[0]
    max_t = t[-1]
    new_N = int(len(t) * resample_rate)
    upsampled_t = np.linspace(min_t, max_t, new_N)
    dt = float(max_t - min_t) / new_N
    return upsampled_t, dt

def SplineExpansion(d, t, n):
    # upsampled_t, dt = resample_time(t)
    beta_n = create_n_order_spline(t, n)
    
    
    return np.convolve(d, beta_n, mode='same')

def q2_a():
    N = 20
    n = 3
    M = 20
    l = np.linspace(-M, (N + 2) * M, N * M + M)
    # t = l / M
    t = np.linspace(-5, 5, 200)
    d = np.random.rand(200)
    x = SplineExpansion(d, t, n)
    
    plt.plot(t, x)
    plt.show()

q2_a()


