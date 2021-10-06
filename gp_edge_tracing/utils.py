import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, median_filter, gaussian_filter
from scipy.ndimage.filters import minimum_filter
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from skimage.measure import shannon_entropy
from skimage import restoration as rest

def kernel_builder(size, b2d=False, normalize=False, vertical_edges=False, unit=False):
    '''
    Based on Lins Code.
    
    This builds a kernel to detect edges vertically (horizontal edges). It return the kernel.
    
    size (tuple) : size of kernel
    b2d (boolean) : True/False value to decide on whether kernel detects bright-to-dark transitions or not.
    normalize (boolean) : True/False value to decide whether to normalize kernel values
    vertical_edges (boolean) : True/False value to transpose kernel so to detect edges horizontally (vertical edges)
    unit (boolean) : True/False to decide whether to have just 1s in the kernel. Used when size has a row/column dimension of 1
    '''
    # Set up kernel dimensions and kernel array
    kernel = np.zeros(size)
    N, M = size
    mid_r = N//2
    mid_c = M//2
    
    # Loop through half the rows to fill with incremental values. If unit=True, then kernel values are all 1 or -1
    if unit:
        for i in range(mid_r):
            kernel[i,:] = 1
            
    else:
        for i in range(mid_r):
            for j in range(M):
                #weight = 1/np.sqrt((i-mid_r)**2+(j-mid_c)**2)
                #weight = max(0, mid_r - np.sqrt((i-mid_r)**2+(j-mid_c)**2) + 1)
                weight = max(0, mid_r + 1 - abs(i-mid_r)-abs(j-mid_c))
                kernel[i,j] = 1 + weight


    # Array values are reflected vertically to fill remaining rows of kernel
    kernel[mid_r+1:,:] = -np.flip(kernel[0:mid_r,:],axis=0)
    
    # If bright-to-dark kernel, flip kernel
    if b2d:
        kernel = np.flipud(kernel)
    
    # To detect vertical edges, take the transpose of kernel
    if vertical_edges:
        kernel = kernel.T
    
    # To normalize values
    if normalize:
        kernel = kernel/kernel.max()

    return kernel



def normalise(img, minmax_val=(0,1), astyp=np.float32):
        '''
        Normalise image between [0, 1]
        
        INPUTS:
        ----------------
            img (2darray) : Image
            
            minmax_val (2-tuple) : Tuple storing minimum and maximum values
            
            astyp (object) : What data type to store normalised image
        '''
        # Extract minimum and maximum values
        min_val, max_val = minmax_val
        
        # Convert to float type
        img = img.astype(np.float32)
        
        # Normalise
        img -= img.min()
        img /= img.max()
        
        # Rescale to max_val
        img *= (max_val - min_val)
        img += min_val
        
        return img.astype(astyp)

    
    
def comp_grad_img(img, kernel):
    '''
    Computes the gradient image using user-defined kernels to measure the horizontal and vertical gradients in both the 
    bright-to-dark and dark-to-bright transitions. Returns and plots the overall gradient image.
    
    INPUTS:
    -----------------
        img (2D-array) : Input image
        kernel (ndarray) : Discrete derivative filter to convolve image with
    '''
    
    # Compute gradient image
    grad_img = convolve(img, kernel, mode='nearest')
    grad_img[np.where(grad_img < 0)] = 0
    grad_img -= grad_img.min()
    grad_img /= grad_img.max()
    
    return grad_img


def plot_imgs(img1, img2, tit1, tit2, size, subplot=True):
    '''
    Plots images with the choice of plotting two in a subplot and adding a title.
    
    img1 (2D-array) : Image 1
    img2 (2D-array) : Image 2
    tit1 (str) : Title for image 1
    tit2 (str) : Title for image 2
    size (tuple): Size of figure
    subplot (bool) : True/False value to plot both image or just one
    '''
    
    if subplot:
        fig = plt.figure(figsize=size)
        ax1 = fig.add_subplot(121)
        ax1.set_title(tit1)
        ax1.imshow(img1, cmap='gray')
        ax2 = fig.add_subplot(122)
        ax2.set_title(tit2)
        ax2.imshow(img2, cmap='gray')
    else:
        fig = plt.figure(figsize=size)
        ax1 = fig.add_subplot(111)
        ax1.set_title(tit1)
        ax1.imshow(img1, cmap='gray')
    
    
def calc_grad_img(img, kd2b, kb2d, plot=True):
    '''
    Computes the gradient image using user-defined kernels to measure the horizontal and vertical gradients in both the 
    bright-to-dark and dark-to-bright transitions. Returns and plots the overall gradient image.
    
    img (2D-array) : Input image
    kd2b (tuple) : Tuple containing the horizontal and vertical dark-to-bright kernels
    kb2d (tuple) : Tuple containing the horizontal and vertical bright-to-dark kernels.
    '''
    # Extract the horizontal and vertical kernels for both dark-to-bright and bright-to-dark transitions
    kd2b_h, kd2b_v = kd2b
    kb2d_h, kb2d_v = kb2d
    
    # Compute horizontal gradient image
    grad_img_d2b_h = convolve(img, kd2b_h, mode='nearest')
    grad_img_d2b_h[np.where(grad_img_d2b_h < 0)] = 0
    grad_img_d2b_h = (grad_img_d2b_h-grad_img_d2b_h.min())/(grad_img_d2b_h.max() - grad_img_d2b_h.min())

    grad_img_b2d_h = convolve(img, kb2d_h, mode='nearest')
    grad_img_b2d_h[np.where(grad_img_b2d_h < 0)] = 0
    grad_img_b2d_h = (grad_img_b2d_h-grad_img_b2d_h.min())/(grad_img_b2d_h.max() - grad_img_b2d_h.min())

    grad_img_h = np.sqrt(grad_img_d2b_h**2 + grad_img_b2d_h**2)
    grad_img_h = (grad_img_h-grad_img_h.min())/(grad_img_h.max() - grad_img_h.min())
    
    
    # Compute vertical gradient image
    grad_img_d2b_v = convolve(img, kd2b_v, mode='nearest')
    grad_img_d2b_v[np.where(grad_img_d2b_v < 0)] = 0
    grad_img_d2b_v = (grad_img_d2b_v-grad_img_d2b_v.min())/(grad_img_d2b_v.max() - grad_img_d2b_v.min())

    grad_img_b2d_v = convolve(img, kb2d_v, mode='nearest')
    grad_img_b2d_v[np.where(grad_img_b2d_v < 0)] = 0
    grad_img_b2d_v = (grad_img_b2d_v-grad_img_b2d_v.min())/(grad_img_b2d_v.max() - grad_img_b2d_v.min())

    grad_img_v = np.sqrt(grad_img_d2b_v**2 + grad_img_b2d_v**2)
    grad_img_v = (grad_img_v-grad_img_v.min())/(grad_img_v.max() - grad_img_v.min())
    
    # Compute overall gradient image
    grad_img = np.sqrt(grad_img_h**2 + grad_img_v**2)
    grad_img = (grad_img-grad_img.min())/(grad_img.max() - grad_img.min())
    
    if plot:
        plot_imgs(img, grad_img, 'Image',  'Kernel Gradient Image',size=(17,15), subplot=True)
    
    return grad_img


def denoise(image, technique, kwargs, plot=False, verbose=False):
    '''This function denoises an image using various algorithms specified by technique.
    
    INPUT:
    ---------
        image (2D Array): Grayscale image to be denoised
            
        technique (str) : Type of denoising algorithm to fit.
        
        kwargs (Dict): Dictionary of parameters for algorithm
    '''    
    if technique == 'nl':
        denoised_img = rest.denoise_nl_means(image, **kwargs)
    elif technique == 'tvc':
        denoised_img = rest.denoise_tv_chambolle(image, **kwargs)
    elif technique == 'wavelet':
        denoised_img = rest.denoise_wavelet(image, **kwargs)
    elif technique == 'tvb':
        denoised_img = rest.denoise_tv_bregman(image, **kwargs)
    elif technique == 'median':
        denoised_img = median_filter(image, **kwargs)
    elif technique == 'gaussian':
        denoised_img = gaussian_filter(image, **kwargs)
    elif technique == 'minimum':
        denoised_img = minimum_filter(image, **kwargs)

    if verbose:
        psnr = round(peak_signal_noise_ratio(image, denoised_img),2)
        ss = round(structural_similarity(image, denoised_img),2)
        nmse = round(normalized_root_mse(image, denoised_img, normalization='min-max'),5)
        entropy = round(shannon_entropy(denoised_img), 3)
        print(f'Peak-SNR: {psnr}.\nStructural Similarity: {ss}.\nMean Square Error: {nmse}.\nShannon Entropy: {entropy}.\n')
        
    if plot:
        plot_imgs(image, denoised_img, 'Image',  'Denoised Image',size=(17,15), subplot=True)
        
    return denoised_img




def construct_test_img(size, amplitude, curvature, noise_level, ltype, intensity, gaps=False):
    '''
    Based on Lins code.
    
    This constructs a test image with a well defined edge(s) with added noise on top. It will return the test image and the
    indexes where the edges are.
    
    INPUTS:
    ----------
        size (tuple) : Size of test image
        
        amplitude (float) : Amplitude of edge
        
        curvature (float) : Frequency of edge
        
        noise_level (float) : Noise variance to add on top
        
        intensity (float) : Intensity of lighter region of image
        
        ltype (str) : Type of edges. Can be straight or sinusoidal and can be basic or complex.
                
        gaps (boolean) : True/False value to add gaps or not to simulate vessel shadows
        
        mode (str) : Type of noise to corrupt test image
    '''
    # Extract test image size, define test_img and set up evenly spaced points to evaluate curve at
    M, N = size    
    test_img = np.zeros((M,N))
    x = np.linspace(-np.pi, np.pi, N)
    if amplitude > M:
        A = M//2
    else:
        A = amplitude//2
    
    # Initialise indexes of edge
    xwave_idx = np.arange(0, N, 1)
    ywave_idx = []
    
    # For each column, evaluate curve, append row index with evaluation and set pixel intensity 
    if ltype=='sinusoidal':
        for j in range(N):
            wave = np.rint(A*np.sin(N*curvature*x[j]))+M//2
            ywave_idx.append(wave.astype('int'))
            test_img[ywave_idx[j]:M, j] = intensity
            
    if ltype=='multi-sinusoidal':
        ywave1_idx = []
        for j in range(N):
            wave = np.rint(A*np.sin(N*curvature*x[j]))+M//2
            ywave_idx.append(wave.astype('int'))
            ywave1_idx.append(ywave_idx[j]+A//2)
            test_img[ywave_idx[j]:M, j] = intensity
            test_img[ywave_idx[j]+A//2:M, j] = 1-intensity
            
    if ltype=='close multi-sinusoidal':
        ywave1_idx = []
        for j in range(N):
            wave = np.rint(A*np.sin(N*curvature*x[j]))+M//2
            ywave_idx.append(wave.astype('int'))
            ywave1_idx.append(ywave_idx[j]+A//6)
            test_img[ywave_idx[j]:M, j] = intensity
            test_img[ywave_idx[j]+A//6:M, j] = 1-intensity
            
    elif ltype=='co-sinusoidal':
        for j in range(N):
            wave = np.rint(A*np.cos(N*curvature*x[j]))+M//2
            ywave_idx.append(wave.astype('int'))
            test_img[ywave_idx[j]:M, j] = intensity
    
    elif ltype=='diag':
        for j in range(N):
            test_img[j:, j] = intensity
            ywave_idx.append(j)
        
    elif ltype=='straight':
        test_img[int(M//2):, :] = intensity
        for j in range(N):
            ywave_idx.append(M//2)
    
    # Store edge indexes
    edge_idx = np.asarray(list(zip(ywave_idx, xwave_idx)))
    
    if ltype=='multi-sinusoidal' or ltype=='close multi-sinusoidal':
        edge_idx = np.concatenate((edge_idx, np.asarray(list(zip(ywave1_idx, xwave_idx)))), axis=0)
            
    # Gaps to simulate vessel shadows
    if gaps:
        test_img[:,20:30] = 0
        test_img[:,N//2:(N//2+10)] = 0
        test_img[:,N-100:N-90] = 0
        test_img[:, N//4:(N//4+20)] = 0
    
    # Add random Gaussian noise
    test_img = random_noise(test_img, mode='gaussian', mean=0, var=noise_level, seed=1)
    
    return test_img, edge_idx