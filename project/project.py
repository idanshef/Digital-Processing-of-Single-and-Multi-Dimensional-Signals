import cv2
import pywt


class WaveletType:
    Daubechies1 = 0

def split_to_patches(img, M, N):
    # TODO: split to patches list/dict?
    pass


def normalize_patches(patches):
    for patch in patches:
        pass
        # TODO: normalize patches to [-1, 1]
        
def wavelet_transform(patches, wavelet_type=WaveletType.Daubechies1):
    # TODO: wavelet type
    # TODO: 3-scale 2D tranform
    wavelet = pywt.Wavelet('db1')
    coeffs = pywt.dwt2(patches, wavelet)
    print("hi")
    pass

def algo(img, M, N):
    # patches = split_to_patches(img, M, N)
    # patches = normalize_patches(patches)
    wavelet_transform(img)

if __name__ == "__main__":
    img_path = r"C:\Users\isheffer\OneDrive - Intel Corporation\Desktop\university\Digital-Processing-of-Single-and-Multi-Dimensional-Signals\finalProject\Lenna.png"
    N, M = 64, 64
    
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    algo(img, M, N)