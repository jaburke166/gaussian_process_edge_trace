# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:31:36 2020

@author: s1522100
"""

import numpy as np
import numpy.matlib as npml
import sklearn.gaussian_process as skgp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import time as t
import sys
sys.path.append('C:\\Users\\s1522100\\Documents\\PhD Work\\Python Scripts\\GP_Edge_Tracing')
import warnings
warnings.filterwarnings('ignore')


#================================= FIT AND PREDICT GAUSSIAN PROCESS =================================
def fit_predict_GP(kernel_params, noise_y, init, N_samples, obs, N_plt_samples, plt_post, size):
    ''' 
    Fits a Gaussian Process to the initial start and end points of the edge and fits observations 
    
    INPUTS:
    ----------
        kernel_params (dict) : Dictionary of parameters for the kernel in the Gaussian Process model. 
                               Will require inputs: 'kernel':'Matern', 'RBF', 'Polynomial', 'Periodic', 'Constant'
                                                    'sigma_f', 'sigma_l' and any kernel-specific parameter.
        
        noise_y (float) : Amount of noise to corrupt non-endpoint observations with.
        
        init (2D-array) : Array of start and end indexes of edge, i.e. init= np.array([[st_x, st_y],[en_x, en_y]])
        
        N_samples (int) : Number of curves to sample from the posterior.
        
        obs (2D-array) : Array of new observations to fit to Gaussian process.
        
        N_plt_samples (int) : Number of samples to plot if displaying posterior distribution.
                
        plt_post (bool) : Flag to plot posterior distribution.
        
        size (tuple) : Size of image
    '''
    # Extract kernel parameters, construct kernel and add to GP parameters
    M, N = size
    bounds = 'fixed'
    sigma_f = kernel_params['sigma_f']
    if kernel_params['kernel']=='RBF':
        sigma_l = kernel_params['sigma_l']
        kernel = sigma_f**2 * skgp.kernels.RBF(sigma_l, length_scale_bounds=bounds)
        
    elif kernel_params['kernel']=='Matern':
        sigma_l, nu = kernel_params['sigma_l'], kernel_params['nu']
        kernel = sigma_f**2 * skgp.kernels.Matern(sigma_l, length_scale_bounds=bounds, nu=nu)
        
    elif kernel_params['kernel']=='Periodic':
        sigma_l, p = kernel_params['sigma_l'], kernel_params['period']
        kernel = sigma_f**2 * skgp.kernels.ExpSineSquared(sigma_l, p, length_scale_bounds=bounds)

    elif kernel_params['kernel']=='Polynomial':
        sigma_l, alpha = kernel_params['sigma_l'], kernel_params['alpha']
        kernel = sigma_f**2 * skgp.kernels.RationalQuadratic(sigma_l, alpha, length_scale_bounds=bounds)

    elif kernel_params['kernel']=='Constant':
        kernel = skgp.kernels.ConstantKernel(sigma_f, bounds)
        
    # Concatenate endpoints and observations and update alpha
    if obs is not None:
        obs = obs.reshape(-1,2)
        new_obs = np.concatenate((init, obs), axis=0)
        X = new_obs[:,0][:, np.newaxis]
        y = new_obs[:,1]
        alpha = np.concatenate((1e-7*np.ones((init.shape[0],)), noise_y*np.ones((obs.shape[0]))), axis=0)
        
    # If first iteration.
    else:
        X = init[:,0][:, np.newaxis]
        y = init[:,1]
        alpha = 1e-7*np.ones((2,))
    
    # Set up Gaussian process model
    gp_params = {'n_restarts_optimizer':0, 'optimizer':None, 'normalize_y':True, 'alpha':alpha, 'kernel':kernel}
    gp = skgp.GaussianProcessRegressor(**gp_params)
    
    # Set up grid of x-axis values
    x_grid = np.linspace(init[:,0][0], init[:,0][1], N)
        
    # Fit data to the GP to update mean and covariance matrix
    gp.fit(X, y)
    
    # Sample N_plt_samples from the posterior  
    y_samples = gp.sample_y(x_grid[:, np.newaxis], N_samples, random_state=None).clip(0, M-1)
    
    # If plotting posterior distribution of GP, predict the GP using range of x-values, plot mean and 67% confidence 
    # deviances from mean. 
    y_mean, y_std = gp.predict(x_grid[:, np.newaxis], return_std=True)
    if plt_post:
        y_plt_samples = gp.sample_y(x_grid[:, np.newaxis], N_plt_samples, random_state=None).clip(0, M-1)
        
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        post_mean = ax.plot(x_grid, y_mean, 'k', lw=3, zorder=9, label='Mean Curve')
        post_sd = ax.fill_between(x_grid, y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='k', label='95% Confidence Region')
        post_obs = ax.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
        _ = ax.plot(x_grid, y_plt_samples, lw=1)
        ax.set_title("Posterior Distribution\n Kernel: {}.".format(kernel),fontsize=20)
        ax.set_xlim([0, N-1])
        ax.set_ylim([M-1, 0])
        ax.set_xlabel('$x$', fontsize=18)
        ax.set_ylabel('$y$', fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        
    return y_mean, y_samples



#================================= INTERPOLATE GRADIENT MAP =================================
def grad_interpolation(grad_img, size, gmin=1e-8):
    '''
    Performs interpolation of gradient values along a curve
    
    INPUTS:
    -----------
        grad_img (2D-array) : Gradient image.
        
        size (tuple): Shape of gradient image

        fill_zero (float) : Values to fill 0-values in gradient image.
    '''
    
    # Extract size of gradient image
    M, N = size
    
    # Create meshgrid of integer points denoting indices of image
    x = np.arange(N)
    y = np.arange(M)
    
    # The surface here is the gradient image. Mark any 0-values to 
    grad_img[grad_img == 0] = gmin
    z = grad_img.T
    
    # Perform multivariate cubic-spline interpolation on the inverse gradient values
    grad_interp = RectBivariateSpline(x,y,z)
    
    return grad_interp



#================================= CALCULATE COST FUNCTION =================================
def cost_funct(grad_interp, new_edge):
    '''
    Computes the line integral along the specified edge given the gradient image. This acts as the cost function that we want
    to optimize.
    
    INPUTS:
    -----------
        grad_interp (func) : Gradient intensity surface to estimate gradient value for off-grid edge indices

        new_edge (2D-array) : edge indexes in xy-space, shape=(N, 2), i.e. new_edge = np.array([[x0, y0],...[xN, yN]])
    '''
    N = new_edge.shape[0]
    x0 = new_edge[0:N-1, 0]
    x1 = new_edge[1:N, 0]
    y_max = np.maximum(new_edge[0:N-1, 1], new_edge[1:N, 1])
    y_min = np.minimum(new_edge[0:N-1, 1], new_edge[1:N, 1])
    
    # Compute the line integral along the edge
    length = 0
    for i in range(new_edge.shape[0]-1):
        length += grad_interp.integral(x0[i], x1[i], y_min[i], y_max[i])
        
    return length



#================================= COMPUTE COSTS AND BEST CURVES =================================
def get_best_curves(grad_interp, y_samples, x_grid, keep_ratio):
    '''
    This function works out the best keep_ratio% curves from the ones sampled from the gaussian process posterior of the 
    previous iteration. It outputs these best curves, the mean curve of these best curves the best curve from all samples.

    INPUTS:
    ----------
        grad_interp (func): Interpolated gradient surface function.

        y_samples (array) : N_sample curves sampled from the gaussian process posterior of the last iteration.
        
        x_grid (array) : Discretised x-axis covering image width using a stepsize of 1.

        keep_ratio (float) : Percentage of best samples to use to determine which points to fix in next iteration.
    '''
    # Store number of samples.
    N_samples = y_samples.shape[1]
    N_keep = int(keep_ratio*N_samples)

    # Stack the samples and their input locations to be fetched for computing their cost
    X = npml.repmat(x_grid, N_samples, 1).T
    curves = np.stack((X, y_samples), axis=2)

    # Initialise list costs of these best curves and compute costs of each best curve
    costs = []
    for i in range(N_samples):
        costs.append(cost_funct(grad_interp, curves[:,i,:]))

    # Select N_keep curves as the best and store optimal curve and its cost
    best_curves = curves[:, np.argsort(costs)[::-1][:N_keep], :]
    optimal_curve = best_curves[:,0,:]
    optimal_cost = max(costs)
    
    # Compute mean curve of these keep_ratio porportion of curves
    # mean_curve = np.mean(best_curves, axis=1)

    return best_curves, (optimal_curve, optimal_cost)



#================================= PLOT BEST CURVES FOR EACH ITERATION =================================

def plot_diagnostics(grad_img, iter_optimal_curves, iter_optimal_costs, size):
    '''
    Plot the best curves superimposed with the true edge, the mean curve from the best curves and current best edge from
    the current iteration.
    
    INPUTS:
    ----------
        grad_img (array) : Gradient image.
    
        iter_optimal_curves (list) : List of optimal curves for each iteration.
        
        iter_optimal_costs (list) : List of costs of optimal curvves for each iteration.

        size (tuple) : Size of image
    '''

    # Plot the most optimal curves from each iteration
    M, N = size
    N_iter = len(iter_optimal_curves)
    fig = plt.figure(figsize=(20,25))
    ax1 = fig.add_subplot(211)
    ax1.imshow(grad_img, cmap='gray')
    for curve in iter_optimal_curves[:-1]:
        ax1.plot(curve[:,1], '--', alpha=0.5)
    ax1.plot(iter_optimal_curves[-1][:,1], '-')
    ax1.legend(['Iteration {}'.format(i+1) for i in range(len(iter_optimal_curves))])
    ax1.set_title('Most optimal curves of each iteration superimposed onto gradient image', fontsize=18)
        
    # Plot costs of the most optimal curves against iteration.
    ax2 = fig.add_subplot(212)
    ax2.scatter(np.arange(1,N_iter+1), iter_optimal_costs, c='r', s=50, edgecolors=(0, 0, 0))
    ax2.set_title('Costs from optimal curves for each iteration', fontsize=18)
    ax2.set_xlabel('Iteration', fontsize=15)
    ax2.set_ylabel('Cost', fontsize=15)
    ax2.set_xticks([i for i in range(1,N_iter+1)])
    
    
    
#================================= COMPUTE GRADIENT IMAGE =================================

def idx_visited(best_curves, size):
    '''
    This function updates the array of visited squares in the gridded search region specified by the bounds and the
    stepsizes delta_x and delta_y
    
    INPUTS:
    ----------
        best_curves (3-Aray) : Best curves from iteration stored in an array of shape: (N, N_samples, 2)

        size (tuple) : Size of image
    '''

    # Extract size of image
    M,N = size

    # If the visisted array isn't specified, initialise a zero array of size (H,W)
    visited_arr = np.zeros((M,N))

    # Given the inputted point, compute the bottom-left vertex of the square that the point exists in  
    i = np.rint(best_curves[:,:,1]).astype('int')
    j = np.rint(best_curves[:,:,0]).astype('int')

    # Update the array index corresponding to the vertex as being visited
    for idx in range(i.shape[1]):
        visited_arr[i[:, idx],j[:,idx]] += 1
        
    return visited_arr



#================================= COMPUTE PIXEL SCORE PER ACCEPTED POINT ================================

def comp_pixel_score(pixel_idx, grad_img, visited_arr, score_thresh, N_samples, pre_fobs_flag=False):
    '''
    This function computes the scores for all the pixels which have had a non-zero count of curve intersections, and then only
    returns those pixels whose score is above the threshold given by score_thresh, in xy-space. 
    
    INPUTS:
    ---------
        pixel_idx (array) : Array of pixel indexes which had a non-zero count of curve intersections from best curves.

        grad_img (array) : Gradient of image.

        visited_arr (array) : Array the same size of the image, storing each pixel indexes curve intersection count.

        score_thresh (float) : Float in [0,1] which decides on the threshold of accepting or rejecting the fixed points.

        N_samples (int)  : Number of samples.

        pre_fobs_flag (bool) : Flag to see if observations being scores are from the previous iteration or from the new 
        iteration.
    '''
    # Compute their gradient value and normalized number of intersections of the best curves
    grad_vals = grad_img[pixel_idx[:,0], pixel_idx[:,1]]
    intersection_vals = (visited_arr[pixel_idx[:,0], pixel_idx[:,1]]/N_samples)
    
    # If computing new scores for previous observations, remove those which have no intersections with best curves of new
    # iteration
    if pre_fobs_flag:
        non_intersection_idx = np.logical_not(intersection_vals==0)
        pixel_idx = pixel_idx[non_intersection_idx]
        grad_vals = grad_vals[non_intersection_idx]
        intersection_vals = intersection_vals[non_intersection_idx]

    # Compute score for each pixel based on gradient value and # of curve intersections. Concatenate best pixels and scores by 
    # thresholding scores.
    pixel_scores = 0.5*(intersection_vals+grad_vals)
    best_scores = pixel_scores[pixel_scores > score_thresh].reshape(-1,1)
    best_pixels = pixel_idx[np.argwhere(pixel_scores > score_thresh)].reshape(-1,2)
    best_pts_scores = np.concatenate((best_pixels[:, [1,0]], best_scores), axis=1)
    
    return best_pts_scores



#================================= BIN POINTS INTO SUB-INTERVALS AND CHOOSE 1 POINT PER SUB-INTERVAL ================================

def bin_pts(N, delta_x, best_pts_scores):
    '''
    This bins the best pixels into sub-intervals splitting up the x-axis and chooses a single pixel index for each sub-interval
    which attains the highest score. These pixel indexes will be fitted to the Gaussian process in xy-space.
    
    INPUTS:
    ----------
    N (int) : The image width.
    
    delta_x (int) : Stepsize to discretize the x-axis from [0, N] creating N//delta_x subintervals to bin the fixed points.
    
    best_pts_scores (array) : Array of points whose score surpasses the score threshold.
    '''
    
    # Bin best points into sub-intervals of length delta_x and choose highest scoring point per sub-interval
    x_intervals = np.arange(0, N, delta_x)
    x_visited = [[] for i in range(x_intervals.shape[0])]
    bin_idx = np.floor(best_pts_scores[:,0]/delta_x).astype(int)
    fobs_scores = []
    for idx, i in enumerate(np.unique(bin_idx)):
        x_visited[i] = best_pts_scores[np.argwhere(bin_idx == i)].reshape(-1,3)
        fobs_scores.append(x_visited[i][np.argmax(x_visited[i][:,2])])
    
    # Convert to array and return the points, not the scores as these are updated in each iteration.
    fobs_scores = np.asarray(fobs_scores)
    fobs = fobs_scores[:,:2].astype(int)
    
    return fobs, fobs_scores



#================================= COMPUTE ACCEPTED FIXED POINTS PER ITERATION =================================

def comp_best_pixels(best_curves, grad_img, score_thresh, delta_x, size, init):
    '''
    This function determines the next set of fixed points to fit to the Gaussian Process
    
    INPUTS:
    ---------
        best_curves (array) :  Best curves from iteration stored in an array of shape: (N, N_samples, 2)

        grad_img (array) : Gradient of image

        score_thresh (float) : Float in [0,1] which decides on the threshold of accepting or rejecting the fixed points

        delta_x (int) : Stepsize to discretize the x-axis from [0, N] creating N//delta_x subintervals to bin the fixed points.
        
        size (tuple) : Size of image
        
        init (array) : Array containing the initial start and end points of edge in xy-space.
    '''
    
    # Determine where in the image the best curves have visited and store these pixel indexes after removing endpoints
    M,N = size
    N_samples = best_curves.shape[1]
    visited_arr = idx_visited(best_curves, size)
    x_st = init[0,0]
    x_en = init[1,0]
    all_pixel_idx = np.argwhere(visited_arr > 0)
    all_pixel_idx = all_pixel_idx[(all_pixel_idx[:,1] != x_st) & (all_pixel_idx[:,1] != x_en)]
    
    # Compute pixel score and threshold pixels using score_thresh
    best_pts_scores = comp_pixel_score(all_pixel_idx, grad_img, visited_arr, score_thresh, N_samples)

    # Bin best points into sub-intervals of length delta_x and choose highest scoring point per sub-interval
    fobs, fobs_scores = bin_pts(N, delta_x, best_pts_scores)
    
    return fobs, fobs_scores, visited_arr



#================================= COMPUTE ITERATION OF EDGE TRACING ALGORITHM =================================

def compute_iter(init, y_samples, x_grid, keep_ratio, score_thresh, delta_x, grad_interp, grad_img, pre_fobs, size, noise_y):
    '''
    This computes an iteration of the edge tracing algorithm. It will compute the cost of all the curves sampled from the 
    gaussian process of the previous iteration. Using only the best keep_curve_ratio of them, a selection of fixed points 
    are accepted to be fitted to the gaussian process model through thresholding using score_thresh, based on each 
    potential fixed points gradient value and number of 'best' curves passing through said fixed points. 
    
    There are also an option reduce the value that is added to the diagonal of the covariance matrix of the gaussian 
    proces to decrease uncertainty (noise) to the observations so that samples from the posterior gaussian process do not 
    always pass through these fixed points.
    
    INPUTS:
    ---------
        init (array) : Array containing the start and last indexes of true edge. Shape: (2,2)

        y_samples (array) : Array of curves sampled from gaussian process of previous iteration. Shape: (N, N_samples)
        
        x_grid (array) : Vector storing the discretized x-axis in the interval [0, N]. Shape: (N, 1)

        keep_ratio (float) : Float between 0 and 1 to denote how many of the best curves to use to determine the fixed
        points for this iteration.

        score_thresh (float) : float between 0 and 1 to denote the threshold between accepting and rejecting fixed points based
        on their score from their gradient image reponse and number of best curve intersections.

        delta_x (int) : Integer to denote break up the interval [0,N] into N//delta_x sub-intervals to avoid fitting fixed 
        points that are too close to one another. If there are multiple fixed points accepted for a particular sub-interval
        then only one point is chosen whose score is the highest.

        grad_interp (func) : Gradient interpolation function for computing costs on the curves.

        grad_img (array) : Gradient image. Shape: (M,N)

        pre_fobs (array) : Array of accepted fixed points from previous iteration. Shape: (N_fixed_pts, 2).
        
        size (tuple) : Size of image.
        
        noise_y (float) : Initial noise variance of new observations. By default is 100.
    '''
    
    # Extract image size, determine number of samples and compute number of sub-intervals.
    M,N = size
    N_samples = y_samples.shape[1]
    N_subintervals = int(N//delta_x)
    
    # Compute best curves and optimal curve
    best_curves, (optimal_curve, optimal_cost) = get_best_curves(grad_interp, y_samples, x_grid, keep_ratio)

    # Determine the set of fixed points which exceed the score threshold for the current iteration. 
    new_fobs, new_fobs_scores, visited_arr = comp_best_pixels(best_curves, grad_img, score_thresh, delta_x, size, init)
    N_new_fobs = new_fobs.shape[0]
    
    # If there are observations from previous observations, store their size to compare with new set of observations.
    N_pre_fobs = 0
    pre_fobs_flag = 0
    if pre_fobs is not None:
        N_pre_fobs = pre_fobs.shape[0]
        pre_fobs_flag = 1
    
    # If the new set of observations is smaller than the previous set (or is too small for first iteration) reduce score and
    # try find more observations.
    while N_new_fobs - N_pre_fobs < 2:
        score_thresh = round(0.95*score_thresh, 4)
        new_fobs, new_fobs_scores, visited_arr = comp_best_pixels(best_curves, grad_img, score_thresh, delta_x, size, init)
        N_new_fobs = new_fobs.shape[0]
    
    # If there are previous observations, recompute the scores for the previous observations and discard those with 
    # no curve intersections of new best curves.
    # Combine the old and new set of observations and bin them accordingly so only one observation is chosen for each 
    # sub-interval.
    all_fobs = new_fobs
    if pre_fobs_flag:
        pre_fobs_scores = comp_pixel_score(pre_fobs[:, [1,0]], grad_img, visited_arr, score_thresh, N_samples, pre_fobs_flag) 
        combine_fobs = np.concatenate((pre_fobs_scores, new_fobs_scores), axis=0)
        all_fobs, _ = bin_pts(N, delta_x, combine_fobs)

    # If there are observations covering 90% of the sub-intervals then reduce alpha. 
    if all_fobs.shape[0] >= 0.9*N_subintervals and (N_pre_fobs < 0.9*N_subintervals):
        noise_y /= 10
        
    return all_fobs, noise_y, score_thresh, (optimal_curve, optimal_cost)



#================================= RUN ENTIRE ALGORITHM TO OUTPUT EDGE =================================

def run_algorithm(init, grad_img, noise_y, N_samples, score_thresh, delta_x, 
                  keep_ratio, kernel_params, print_diagnostics=False):
    '''
    Outer function that runs the Gaussian process edge tracing algorithm. The inputs are chosen by the user
    
    INPUTS:
    -----------
        init (2D-array) : Array of start and end indexes of edge, i.e. init= np.array([[st_x, st_y],[en_x, en_y]]).

        grad_img (array) : Gradient of image

        noise_y (float) : Amount of noise to perturb the samples drawn from Gaussian process from the non-endpoint observations.
    
        N_samples (int) : Number of samples to draw from Gaussian process
        
        score_thresh (float) : Initial pixel score to determine which observations to accept and fit to the Gaussian process in
        each iteration.
        
        delta_x  (int) : Length of sub-interval to split x-axis into.
        
        keep_ratio (float) : Proportion of best curves to use to score and accept observations.
        
        kernel_params (dict) : Dictionary of parameters for the kernel in the Gaussian Process model. 
                               Will require inputs: 'kernel':'Matern', 'RBF', 'Polynomial', 'Periodic', 'Constant'
                                                    'sigma_f', 'sigma_l' and any kernel-specific parameter.
                                                    
        print_diagnostics (bool) : Flag to plot the optimal curves and their costs on the gradient image.
    '''
    
    # Extract size of image, compute interpolated gradient intensity function and compute the x-component pixel indexes for
    # the length of the edge dictated by init
    size = grad_img.shape
    N,M = size
    grad_interp = grad_interpolation(grad_img, size)
    x_grid = np.linspace(init[0,0], init[1,0], init[1,0]-init[0,0]+1).astype(int)
    
    # Initialise the number of sub-intervals dividing the x-axis, the number of observations added to the Gaussian process
    # and the two lists storing the optimal curve and its cost in each iteration
    N_subintervals = int(N//delta_x)
    pre_fobs = None
    n_fobs = 0
    iter_optimal_curves = []
    iter_optimal_costs = []
    
    # Convergence of the algorithm is when there is an observation fitted for each sub-interval
    N_iter = 1
    while n_fobs < N_subintervals:
        # Start timer for iteration
        st = t.time()
        
        # Intiialise Gaussian process by fitting initial points, outputting samples drawn from initial posterior distribution 
        _, y_samples = fit_predict_GP(kernel_params, noise_y, init, N_samples, pre_fobs, 1, False, size)
        
        # Compute a single iteration of the algorithm to output new set of observations, new values of noise_y and score_thresh
        # as well as the optimal curve and its cost to be appended to separate lists for plotting at the end of the algorithm.
        output = compute_iter(init, y_samples, x_grid, keep_ratio, score_thresh, delta_x, 
                              grad_interp, grad_img, pre_fobs, size, noise_y)
        pre_fobs, noise_y, score_thresh, (optimal_curve, optimal_cost) = output
        iter_optimal_curves.append(optimal_curve)
        iter_optimal_costs.append(optimal_cost)
        
        # Recompute number of observations
        n_fobs = pre_fobs.shape[0]
        
        # Incremement number of iterations and output time taken to perform iteration
        en = t.time()
        print('Iteration {} - Time Elapsed: {}\n'.format(N_iter, round(en-st, 4)))
        N_iter += 1
     
    # When convergence is reached, option to plot the optimal curves on the gradient image and their costs on a separate plot.
    if print_diagnostics:
        plot_diagnostics(grad_img, iter_optimal_curves, iter_optimal_costs, size)
    
    # When convergence is reached, trial several different values of alpha and choose alpha which produces highest scoring 
    # mean curve. Output this optimal mean curve in pixel space as the trace of the edge.
    alpha_costs = []
    alpha_curves = []
    potential_alpha = [10**(-i) for i in range(-1, 10)]
    for alpha in potential_alpha:
        noise_y = alpha
        try:
            y_mean, _ = fit_predict_GP(kernel_params, noise_y, init, N_samples, pre_fobs, 1, False, size)
            mean_curve = np.concatenate((x_grid[:,np.newaxis], y_mean[:, np.newaxis]), axis=1)
            alpha_curves.append(mean_curve)
            alpha_costs.append(cost_funct(grad_interp, mean_curve))  
        except:
            pass
    edge_trace = alpha_curves[np.argmax(alpha_costs)][:, [1,0]]
    
    return edge_trace