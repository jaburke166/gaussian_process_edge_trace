import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve, median_filter, gaussian_filter, minimum_filter
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from skimage.measure import shannon_entropy
from skimage import restoration as rest

def kernel_builder(size, b2d=False, normalize=False, vertical_edges=False, unit=False):
    '''
    This builds a kernel to detect edges edges in an image. This is effectively an extension to the Sobel operator.
    
    INPUTS:
    --------------
        size (tuple) : size of kernel
        
        b2d (bool, default False) : If flagged, kernel will output kernel suitable for detecting bright-to-dark intensity transitions.

        normalize (bool, default False) : If flagged, kernel will be normalised to (0,1).

        vertical_edges (bool, default False) : If flagged, kernel will detect vertical edges rather than horizontal ones by default.

        unit (bool, default False) : If flagged, non-zero values will only contain 1.
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

    
    
def comp_grad_img(img, kernel, norm=True, astyp=np.float32):
    '''
    Computes the gradient image using user-defined kernels to measure the horizontal and vertical gradients in both the 
    bright-to-dark and dark-to-bright transitions.
    
    INPUTS:
    -----------------
        img (2darray) : Input image.
        
        kernel (2darray) : Discrete derivative filter to convolve image with.
        
        norm (bool, default True) : If flagged, image gradient is normalised to (0,1).
        
        astyp (data-type, default np.float32) : Which type to set image data as
    '''
    
    # Compute gradient image
    grad_img = convolve(img, kernel, mode='nearest')
    grad_img[np.where(grad_img < 0)] = 0
    if normalise:
        output = normalise(grad_img, minmax_val=(0, 1), astyp=astyp)
    else:
        output = grad_img.astype(int)
    
    return output


def denoise(image, technique, kwargs, plot=False, verbose=False):
    '''This function denoises an image using various algorithms specified by technique.
    
    INPUT:
    ---------
        image (2darray): Grayscale image to be denoised
            
        technique (str) : Type of denoising algorithm to fit.
        
        kwargs (dict): Dictionary of parameters for algorithm
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
    else:
        print("Denoising technique not implemented.")
        denoised_img = None

    if verbose:
        psnr = round(peak_signal_noise_ratio(image, denoised_img),2)
        ss = round(structural_similarity(image, denoised_img),2)
        nmse = round(normalized_root_mse(image, denoised_img, normalization='min-max'),5)
        entropy = round(shannon_entropy(denoised_img), 3)
        print(f'Peak-SNR: {psnr}.\nStructural Similarity: {ss}.\nMean Square Error: {nmse}.\nShannon Entropy: {entropy}.\n')
        
    return denoised_img




def construct_test_img(size, amplitude, curvature, noise_level, ltype, intensity, gaps=False):
    '''    
    This constructs a test image with a well defined edge(s) with added noise on top. 
    
    INPUTS:
    ----------
        size (tuple) : Size of test image.
        
        amplitude (float) : Amplitude of edge.
        
        curvature (float) : Frequency of edge.
        
        noise_level (float) : Noise variance to add on top.
        
        intensity (float) : Intensity of lighter region of image.
        
        ltype (str) : Type of edges. Can be straight or sinusoidal and can be basic or complex.
                
        gaps (bool, default False) : If flagged, add gaps to occlude parts of edge of interest.
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


def trace_MSE(edge_pred, edge_true):
    '''
    Return the mean squared error between true edge and edge prediction.
    
    INPUTS:
    ---------------
        edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
        
        edge_true (2darray) : Ground truth array of edge of interest.
    '''
    N = edge_pred.shape[0]
    if edge_pred.ndim == 1:
        edge_pred = edge_pred.reshape(-1,1)
    return np.round((1/N) * np.sum((edge_pred[:,0] - edge_true[:,0])**2),4)

def trace_relarea(edge_pred, edge_true):
    '''
    Return the relative change in area dictated by ground truth edge and edge prediction. This is equivalent to intersection-over-union.
    
    INPUTS:
    ---------------
        edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
        
        edge_true (2darray) : Ground truth array of edge of interest.
    '''
    N = edge_pred.shape[0]
    if edge_pred.ndim == 1:
        edge_pred = edge_pred.reshape(-1,1)
    true_area = (np.sum(N - edge_true[:,0]) / N**2)
    pred_area = (np.sum(N - edge_pred[:,0]) / N**2)
    return np.round(np.abs((true_area-pred_area) / true_area), 5)

def trace_dicecoef(edge_pred, edge_true, jaccard=False):
    '''
    Return the DICE similarity coefficient between the edge prediction and ground truth.
    
    INPUTS:
    ---------------
        edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
        
        edge_true (2darray) : Ground truth array of edge of interest.
        
        jaccard (bool) : If flagged, return Jaccard index instead of DICE (J = D / (2-D) or D = 2J / (J+1))
    '''
    N = edge_pred.shape[0]
    if edge_pred.ndim == 1:
        edge_pred = edge_pred.reshape(-1,1)
    pred_bin = np.zeros((N,N))
    true_bin = np.zeros_like(pred_bin)
    for i in range(N):
        pred_bin[int(edge_pred[i,0]):, i] = 1
        true_bin[int(edge_true[i,0]):, i] = 1
    jacc = (np.sum(pred_bin*true_bin)/np.sum(np.clip((pred_bin+true_bin), 0, 1)))
    
    if jaccard:
        return np.round(jacc, 4)
    else:
        return np.round((2*jacc / (jacc+1)), 4)

def plot_results(edge_trace, true_edge, test_img, grad_img, credint=None, string='True Edge vs. Edge Pred'):
    '''
    Plot results from edge trace onto test_img, grad_img.
    
    INPUTS:
    ---------------
        edge_pred (2darray) : Array storing pixel coordinates (in yx-space) of edge prediction.
        
        edge_true (2darray) : Ground truth array of edge of interest.
        
        test_img (2darray) : Test image.
        
        grad_img (2darray) : Image gradient.
        
        credint (bool, default None) : Object to store 95% credible interval of edge prediction.
        
        string (str, default 'True Edge vs. Edge Pred') : String to display on axis 1.
    '''
    # Compute metrics
    if edge_trace.ndim == 1:
        edge_trace = edge_trace.reshape(-1,1)
    rel_area_diff = trace_relarea(edge_trace, true_edge)
    dice_coeff = trace_dicecoef(edge_trace, true_edge)
    mse = trace_MSE(edge_trace, true_edge)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,8))
    ax1.imshow(test_img, cmap='gray')
    ax1.set_title(f'{string}', fontsize=20)
    ax2.imshow(grad_img, cmap='gray')
    ax2.set_title(f'MSE: {mse}, Rel. Area Diff: {rel_area_diff}, DICE: {dice_coeff}', fontsize=20)
    ax1.plot(true_edge[[0,-1],1], true_edge[[0,-1],0], 'o', c='r', markersize=5, label='Edge Endpoints')
    ax2.plot(true_edge[[0,-1],1], true_edge[[0,-1],0], 'o', c='r', markersize=5, label='Edge Endpoints')
    ax1.plot(true_edge[:,1], edge_trace[:,0], 'r-', zorder=2, label='Proposed')
    ax1.plot(true_edge[:,1], true_edge[:,0], 'b--', linewidth=2, label='Ground Truth')
    ax2.plot(true_edge[:,1], edge_trace[:,0], 'r-', zorder=2, label='Proposed')
    ax2.plot(true_edge[:,1], true_edge[:,0], 'b--', linewidth=2, label='Ground Truth')
    if credint is not None:
        ax1.fill_between(true_edge[:,1], credint[0], credint[1], alpha=0.5, 
                         color='m', zorder=1, label='95% Credible Region')
        ax2.fill_between(true_edge[:,1], credint[0], credint[1], alpha=0.5, 
                         color='m', zorder=1, label='95% Credible Region')
    ax1_legend = ax1.legend(ax1.get_legend_handles_labels()[1], fontsize=13, ncol=2, 
                      loc='lower right', edgecolor=(0, 0, 0, 1.))
    ax1_legend.get_frame().set_alpha(None)
    ax1_legend.get_frame().set_facecolor((1,1,1,1))
    ax2_legend = ax2.legend(ax2.get_legend_handles_labels()[1], fontsize=13, ncol=2, 
                      loc='lower right', edgecolor=(0, 0, 0, 1.))
    ax2_legend.get_frame().set_alpha(None)
    ax2_legend.get_frame().set_facecolor((1,1,1,1))
    fig.tight_layout()
    
    return fig