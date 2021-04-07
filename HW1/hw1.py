import numpy as np
import matplotlib.pyplot as plt
from numpy.fft.helper import fftshift

def compute_g(time, tau=1.):
    u = np.zeros(len(time))
    u[time >= 0] = 1

    return (1./tau) * np.exp(-time / tau) * u


def q1():
    N = 1000
    t_range = 5
    t = np.linspace(-t_range, t_range, N)
   
    a = np.random.normal(0., 1., N)
    tau = 1.
    g = compute_g(t, tau)
    x = tau * np.convolve(a, g, mode='same')

    x_freq = np.abs(np.fft.fft(x))
    sample_freq = np.fft.fftfreq(N, d=(2 * t_range) / N)
    
    plt.figure()
    plt.subplot(211)
    plt.title(r'$a_n$')
    plt.xlabel('n')
    plt.plot(a)
    plt.subplot(212)
    plt.title(r"$\|X(e^{2\pi jf})\|$")
    plt.xlabel('f')
    plt.plot(sample_freq, x_freq)
    plt.show()
    
    
    g = np.sinc(t * 1e20)
    new_a = np.convolve(x, g, mode='same')
    # new_a = np.fft.fft(x)
    new_x = np.convolve(new_a, g, mode='same')
    plt.figure()
    plt.subplot(211)
    plt.title(r'$x(t) = \sum a_n g(t-n)$')
    plt.xlabel('t')
    plt.plot(t, x)
    plt.subplot(212)
    plt.title(r'$x(t) = \sum \tilde{a_n} sinc(t-n)$')
    plt.xlabel('t')
    plt.plot(t, new_x)
    plt.show()

q1()