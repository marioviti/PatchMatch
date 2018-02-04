from patchmatch import nnf_approx
from skimage.io import imread
import numpy as np
import time
from matplotlib import pyplot as plt

def ssd(patch1, patch2):
    return np.sum((patch1-patch2)**2)

def is_clamped(h,w,i0,i1,j0,j1):
    p1 = i0 < h and i0 > 0
    p2 = i1 < h and i1 > 0
    p3 = j0 < w and j0 > 0
    p4 = j1 < w and j1 > 0
    return p1 and p2 and p3 and p4

def reconstruction(A,B,nnf,patch_size):
    reconsturcted = np.zeros(A.shape)
    weights = np.ones(A.shape)
    nnf_h,nnf_w = nnf.shape[0:2]
    a_h, a_w = A.shape[0:2]
    b_h, b_w = B.shape[0:2]
    for i in range( patch_size//2+1, nnf_h):
        for j in range( patch_size//2+1, nnf_w):
            nnf_i = nnf[i,j,0]
            nnf_j = nnf[i,j,1]
            for k in range(-patch_size//2,1):
                for l in range(-patch_size//2,1):
                    i1,j1 = i+k,j+l
                    nnf_i1,nnf_j1 = int(nnf_i)+l,int(nnf_j)+k
                    # check if the patch is in the area of B
                    if is_clamped( b_h, b_w, nnf_i1, nnf_i1+patch_size, nnf_j1, nnf_j1+patch_size ):
                        reconsturcted[i1:i1+patch_size, j1:j1+patch_size] += \
                            B[nnf_i1:nnf_i1+patch_size, nnf_j1:nnf_j1+patch_size]
                        weights[i1:i1+patch_size,j1:j1+patch_size] += 1.0
    return reconsturcted/weights


def initialization(A,B,distance_function,patch_size):
    h,w = A.shape[0:2]
    h_nnf = h-patch_size//2*2
    w_nnf = w-patch_size//2*2
    nnf = np.zeros((h_nnf,w_nnf,3))
    for i in range(h_nnf):
        for j in range(w_nnf):
            # initialize with random values and calculate distance
            nnf_i = np.random.randint(0,high=h_nnf)
            nnf_j = np.random.randint(0,high=w_nnf)
            nnf[i,j,0] = float(nnf_i)
            nnf[i,j,1] = float(nnf_j)
            nnf[i,j,2] = distance_function( A[i:i+patch_size, j:j+patch_size],
                            B[nnf_i:nnf_i+patch_size, nnf_j:nnf_j+patch_size])
    return nnf

def main():
    np.random.seed(int((time.time()*1e6)%1e6))
    path_image_a = './a.png'
    path_image_b = './b.png'
    A = imread(path_image_a)/255.
    B = imread(path_image_b)/255.
    ps = 5
    nnf = initialization(A,B,ssd,ps)
    print('in')
    #nnf1 = nnf_approx(A,B,nnf,ps,5)
    print('out')
    #rec2 = reconstruction(A,B,nnf1,ps)
    rec2 = nnf_approx(A,B,nnf,ps,5)
    print(np.unique(rec2))
    plt.imshow(rec2)
    plt.show()

if __name__ == '__main__':
    main()
