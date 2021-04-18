import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import cv2


def create_beta_zero(t):
    beta_0 = np.zeros_like(t)
    beta_0[np.bitwise_and(-0.5 < t, t < 0.5)] = 1.
    beta_0[np.bitwise_or(t == 0.5, t == -0.5)] = 0.5
    return beta_0

def create_n_order_spline(t, n):
    assert 0 <= n <= 3, "n must be 0 <= n <= 3"
    beta_0 = create_beta_zero(t)
    if n == 0:
        return beta_0

    delta_t = t[1] - t[0]
    beta_n = np.copy(beta_0)
    
    for i in range(1, n + 1):
        beta_n = delta_t * np.convolve(beta_n, beta_0)
    
    return beta_n

def SplineExpansion(d, t, n):
    N = len(d)
    delta_t = t[1] - t[0]

    beta_0 = np.bitwise_and(t < 1/2, t > -1/2).astype(int) + 1/2 * np.bitwise_or(t == 1/2, t == -1/2).astype(int)
    beta = beta_0.copy()

    for j in range(n):
        beta = np.convolve(beta, beta_0) * delta_t
        
    beta_shift_0 = beta[beta!=0]
    n = len(t)
    nn = len(beta_shift_0)
    n_zeros = int(np.floor(n/2)- np.floor(nn/2))
    gap = 2*n_zeros + len(beta_shift_0) - len(t)
    if gap > 0:
        beta_shift_0 = np.hstack((np.zeros(n_zeros-1, dtype=float), beta_shift_0, np.zeros(n_zeros, dtype=float)))
    elif gap < 0:
        beta_shift_0 = np.hstack((np.zeros(n_zeros+1, dtype=float), beta_shift_0, np.zeros(n_zeros, dtype=float)))
    else:
        beta_shift_0 = np.hstack((np.zeros(n_zeros, dtype=float), beta_shift_0, np.zeros(n_zeros, dtype=float)))
    
    max_ind = np.argmax(beta_shift_0)
    beta_shift_0 = np.roll(beta_shift_0, len(beta_shift_0) - max_ind)
        
    x = 0.
    for i in range(N):
        beta_shift = np.roll(beta_shift_0, int(round(i * 1/delta_t, 5)))
        x = x + d[i]* beta_shift
    
    return x


def derivativeSpline(d, t, n):
    spline = SplineExpansion(d, t, n-1)
    x1 = np.roll(spline, 10)
    x2 = np.roll(spline, -10)
    return x2 - x1


def interpCubic(c, t):
    d_cas = lfilter(np.array([2-np.sqrt(3)]), np.array([1, 2-np.sqrt(3)]), c)
    d = lfilter(np.array([6]), np.array([1, 2-np.sqrt(3)]), d_cas[::-1])
    
    return SplineExpansion(d[::-1], t, 3)


def q2_2():
    N = 100
    n = 3
    M = 20
    l = np.linspace(-M, (N + 2) * M, (N + 3) * M + 1)
    d = np.random.rand(N) - 0.5
    x = SplineExpansion(d, l / M, n)
    
    X_derivative = derivativeSpline(d, l / M, n)
    
    delta_t = (l[1] - l[0]) / M
    X_diff = np.diff(np.hstack((x,0))) / delta_t
    
    fig, axs = plt.subplots(2)
    
    axs[0].set_title("X(t)")
    axs[0].set_ylabel("Gain")
    axs[0].set_xlabel("t")
    axs[0].plot(l, x)
    
    axs[1].set_title("First derivative approximation")
    axs[1].set_ylabel("Gain")
    axs[1].set_xlabel("t")
    axs[1].plot(l, X_derivative, linestyle="--", label="Splines")
    axs[1].plot(l, X_diff, label="Numeric")
    
    axs[1].legend()
    
    # plt.show()


def q2_4():
    N = 100
    M = 20
    l = np.linspace(-M, (N + 2) * M, (N + 3) * M + 1)
    d = np.random.rand(N) - 0.5
    
    x = SplineExpansion(d, l / M, 3)

    c = x[::M]
    x_rec = interpCubic(c, l / M)
    
    plt.figure()
    plt.title("X(t)")
    plt.ylabel("Gain")
    plt.xlabel("t")
    plt.plot(l, x, linestyle="--", label="Original")
    plt.plot(l, x_rec, label="Reconstructed")
    
    plt.legend()
    # plt.show()


def q2_5():
    M=20
    n = np.linspace(-10, 10, 21)
    t = np.linspace(-10, 10, 20 * M + 1)
    delta = (n==0).astype(int)
    cardinal = interpCubic(delta, t)
    
    plt.figure()
    plt.title("Cardinal - sinc")
    plt.xlabel("t")
    plt.ylabel("Gain")
    plt.plot(t, cardinal, label="cardinal")
    plt.plot(t, np.sinc(t), label="sinc")
    
    plt.legend()
    # plt.show()


def q2_6():
    img = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
    reconstructed = np.zeros_like(img, dtype=float)
    (N, M, k) = img.shape
    img = img / 255.
    img_downsampled = cv2.resize(img, (int(M/2), int(N/2)), interpolation=cv2.INTER_NEAREST)

    t = np.arange(M)
    for channel in range(k):
        for row in range(int(N/2)):
            coef = np.sqrt(8) - 3
            c_local = lfilter(np.array([8]), np.array([1, -coef]), img_downsampled[row, :, channel])
            c_filtered = lfilter(np.array([-coef]), np.array([1, -coef]), c_local[::-1])
    
            reconstructed[row, :, channel] = SplineExpansion(c_filtered[::-1], t/2, 2)

    t2 = np.arange(N)
    for channel in range(k):
        for column in range(M):
            coef = np.sqrt(8) - 3
            c_local_col = lfilter(np.array([8]), np.array([1, -coef]), reconstructed[:, column, channel])
            c_filtered_col = lfilter(np.array([-coef]), np.array([1, -coef]), c_local_col[::-1])
            
            reconstructed[:, column, channel] = SplineExpansion(c_filtered_col[::-1], t2/2, 2)
    
    reconstructed = (reconstructed - reconstructed.min())
    reconstructed = reconstructed / reconstructed.max()
    
    fig, axs = plt.subplots(1,3)
    
    axs[0].set_title("Original")
    axs[0].imshow(img)
    
    axs[1].set_title("Reconstructed")
    axs[1].imshow(reconstructed)
    
    axs[2].set_title("Difference")
    axs[2].imshow(img - reconstructed)
    
    # plt.show()
    

def q2_7():
    img = cv2.imread('img.jpg')
    
    (N, M, k) = img.shape
    img = img / 255.
    upsampled = np.zeros((2*N, 2*M , k))

    t = np.arange(2*M)
    for channel in range(k):
        for row in range(N):
            c = img[row, :, channel]
            
            coef = np.sqrt(3) - 2
            c_local = lfilter(np.array([6]), np.array([1, -coef]), c)
            c_filtered = lfilter(np.array([-coef]), np.array([1, -coef]), c_local[::-1])
            
            upsampled[row, :, channel] = SplineExpansion(c_filtered[::-1], t/2, 3)

    t2 = np.arange(2*N)
    for channel in range(k):
        for column in range(2*M):
            c_col = upsampled[0:N, column, channel]
            
            coef = np.sqrt(3) - 2
            c_local = lfilter(np.array([6]), np.array([1, -coef]), c_col)
            c_filtered_col = lfilter(np.array([-coef]), np.array([1, -coef]), c_local[::-1])
            
            upsampled[:, column, channel] = SplineExpansion(c_filtered_col[::-1], t2/2, 3)
            
    downsampled_temp = np.zeros((2*N,M,k))
    downsampled = np.zeros_like(img)

    t = np.arange(M)
    for channel in range(k):
        for row in range(2*N):
            c = upsampled[row, :, channel]
            downsampled_temp[int(np.ceil(row/2)), :, channel] = SplineExpansion(c, 2*t, 1)

    t2 = np.arange(N)
    for channel in range(k):
        for column in range(M):
            c_col = downsampled_temp[:, column, channel]
            downsampled[:, column, channel] = SplineExpansion(c_col, t2, 1)


    upsampled = (upsampled - upsampled.min())
    upsampled = upsampled / upsampled.max()
    
    downsampled = (downsampled - downsampled.min())
    downsampled = downsampled / downsampled.max()
    
    fig, axs = plt.subplots(2,2)
    
    axs[0,0].set_title('Original image')
    axs[0,0].imshow(img)
    
    axs[0,1].set_title('Upsampled - spline order 3')
    axs[0,1].imshow(upsampled)
    
    axs[1,0].set_title('Downsampled - spline order 1')
    axs[1,0].imshow(downsampled)
    
    axs[1,1].set_title('Difference')
    axs[1,1].imshow(img - downsampled)
    
    


q2_2()
q2_4()
q2_5()
q2_6()
q2_7()

plt.show()