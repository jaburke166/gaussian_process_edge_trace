# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:01:44 2021

@author: s1522100
"""

import numpy as np
import numpy.matlib as npml
import sklearn.gaussian_process as skgp
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import simps, trapz
import time as t
import sys
from io import StringIO


#================================= FIT AND PREDICT GAUSSIAN PROCESS =================================
def fit_predict_GP(kernel_params, noise_y, init, size, N_samples=1000, obs=np.array([]), seed=1522100, 
                   plt_prior_post=False, N_plt_samples=20, init_noise_y=1e-7):
    ''' 
    Fits a Gaussian Process to the initial start and end points of the edge and fits observations 
    
    INPUTS:
    ----------
        kernel_params (dict) : Dictionary of parameters for the kernel in the Gaussian Process model. 
                               Will require inputs: 'kernel':('Matern', 'RBF', 'Polynomial'), 'sigma_f', 'sigma_l'
                                                     and any kernel-specific parameter.
        
        noise_y (float) : Amount of noise to corrupt non-endpoint observations with.
        
        init (2D-array) : Array of start and end indexes of edge, i.e. init= np.array([[st_x, st_y],[en_x, en_y]]).
        
        size (tuple) : Size of image.
        
        N_samples (int) : Number of curves to sample from the posterior. Default value: 1000.
        
        obs (2D-array) : Array of new observations to fit to Gaussian process. If at first iteration, this should  be
                         an empty array, i.e. obs=np.array([]), unless propagating points. Default value: np.array([]).
        
        seed (int) : Seed tp fix random sampling of gaussian process. Default value: 1522100
        
        plt_prior_post (bool) : Flag to plot prior and posterior distribution using N_plt_samples curves sampled from 
        Gaussian process. Default value: False.
        
        N_plt_samples (int) : Number of curves to sample from Gaussian process to sample if plt_post=True. 
        Default value: 20.
        
        init_noise_y (float) : Initial noise to perturb the edge endpoints. Only usually changed for propagating points from
        a previous edge to the current edge of interest. Default value: 1e-7.
     '''
    # Extract kernel parameters, construct kernel and add to GP parameters
    M, N = size
    bounds = 'fixed'
    k_params = kernel_params.copy()
    sigma_f = k_params['sigma_f']**2
    kernel_type = k_params['kernel']
    del k_params['sigma_f'],  k_params['kernel']

    if kernel_type == 'Matern':
        kernel = sigma_f * skgp.kernels.Matern(**k_params, length_scale_bounds=bounds)
    elif kernel_type == 'RBF':
        kernel = sigma_f * skgp.kernels.RBF(**k_params, length_scale_bounds=bounds)
    elif kernel_type == 'Polynomial':
        kernel = sigma_f * skgp.kernels.RationalQuadratic(**k_params, length_scale_bounds=bounds)
            
    # Concatenate endpoints and observations and update alpha. Also perturb initial pixel locations if their y-values are
    # identical as this causes an error internally when these are the only 2 points being fitted to the GP.
    obs = obs.reshape(-1,2)
    new_obs = np.concatenate((init, obs), axis=0)
    X = new_obs[:,0][:, np.newaxis]
    y = new_obs[:,1]
    y[0], y[1] = y[0]-0.25, y[1]+0.25
    alpha = np.concatenate((init_noise_y*np.ones((init.shape[0],)), noise_y*np.ones((obs.shape[0]))), axis=0)

    # Set up Gaussian process model
    gp_params = {'normalize_y':True, 'alpha':alpha, 'kernel':kernel,
                'n_restarts_optimizer':0, 'optimizer':None, 'copy_X_train':True,
                'random_state':1}
    gp = skgp.GaussianProcessRegressor(**gp_params)
    
    # Set up grid of x-axis values
    x_grid = np.linspace(init[:,0][0], init[:,0][-1], N)
    
    if plt_prior_post:
        y_mean_prior, y_std_prior = gp.predict(x_grid[:, np.newaxis], return_std=True)
        y_plt_samples_prior = gp.sample_y(x_grid[:, np.newaxis], N_plt_samples, random_state=seed).clip(-M//2, M//2)
        
    # Fit data to the GP to update mean and covariance matrix
    gp.fit(X, y)
    
    # Sample N_plt_samples from the posterior  
    y_samples = gp.sample_y(x_grid[:, np.newaxis], N_samples, random_state=seed).clip(0, M-1)
    
    # If plotting posterior distribution of GP, predict the GP using range of x-values, plot mean and 67% confidence 
    # deviances from mean. 
    y_mean, y_std = gp.predict(x_grid[:, np.newaxis], return_std=True)
    
    if plt_prior_post:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,20))
        prior_mean = ax1.plot(x_grid, y_mean_prior, 'k', lw=3, zorder=9, label='Mean Curve')
        prior_sd = ax1.fill_between(x_grid, y_mean_prior - 2*y_std_prior, y_mean_prior + 2*y_std_prior, 
                                    alpha=0.2, color='k', label='95% Confidence Region')
        _ = ax1.plot(x_grid, y_plt_samples_prior, lw=1)
        ax1.set_title("Prior Distribution\n Kernel: {}".format(kernel), fontsize=20)
        ax1.set_xlim([0, N-1])
        ax1.set_xlabel('$x$', fontsize=18)
        ax1.set_ylabel('$y$', fontsize=18)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        
        y_plt_samples_post = gp.sample_y(x_grid[:, np.newaxis], N_plt_samples, random_state=seed).clip(0, M-1)
        
        post_mean = ax2.plot(x_grid, y_mean, 'k', lw=3, zorder=9, label='Mean Curve')
        post_sd = ax2.fill_between(x_grid, y_mean - 2*y_std, y_mean + 2*y_std, 
                                   alpha=0.2, color='k', label='95% Confidence Region')
        post_obs = ax2.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
        _ = ax2.plot(x_grid, y_plt_samples_post, lw=1)
        ax2.set_title("Posterior Distribution\n Kernel: {}.".format(kernel),fontsize=20)
        ax2.set_xlim([0, N-1])
        ax2.set_ylim([M-1, 0])
        ax2.set_xlabel('$x$', fontsize=18)
        ax2.set_ylabel('$y$', fontsize=18)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels)
        
        fig.show()
        plt.pause(0.5)
        
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
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()]).T
    
    # The surface here is the gradient image. Mark any 0-values to gmin
    grad_img[grad_img == 0] = gmin
    Z = grad_img.ravel()
    
    # Perform basic linear 2-dimensional interpolation to create gradient surface
    grad_interp = LinearNDInterpolator(XY, Z)
    
    return grad_interp



#================================= CALCULATE COST FUNCTION =================================
def cost_funct(grad_interp, edge):
    '''
    Computes the line integral along the specified edge given the gradient image. This acts as the cost function that we want
    to optimize.
    
    INPUTS:
    -----------
        grad_interp (func) : Gradient intensity surface to estimate gradient value for off-grid edge indices

        edge (2D-array) : edge indexes in xy-space, shape=(N, 2), i.e. new_edge = np.array([[x0, y0],...[xN, yN]])
    '''
    # Evaluate edge along interpolated gradient image 
    edge = edge[edge[:,0].argsort(), :]
    grad_score = grad_interp(edge)
    
    # Compute cumulative sum of euclidean distance between pixel indexes of edge.
    # This is the equivalent of computing the curvilinear coordinates of the edge
    pixel_diff = np.cumsum(np.sqrt(np.sum(np.diff(edge, axis=0)**2, axis=1)))
                           
    # Compute line integral
    line_integral = 1/trapz(grad_score[:-1], pixel_diff)
                           
    return line_integral
                           


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
    costs = np.asarray(costs)
       
    # Select N_keep curves as the best curves and their costs
    best_idxs = np.argsort(costs)[:N_keep]
    best_curves = curves[:, best_idxs, :]
    best_costs = costs[best_idxs]
    #print(best_costs)
    
    # Store optimal curve and its cost
    optimal_curve = best_curves[:,0,:]
    optimal_cost = best_costs[0]

    return best_curves, best_costs, (optimal_curve, optimal_cost)



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
    for i, curve in enumerate(iter_optimal_curves[:-1]):
        ax1.plot(curve[:,1], '--', alpha=0.25, label='Iteration {}'.format(i+1))
    ax1.plot(iter_optimal_curves[-1][:,1], '-', label='Final Edge')
    ax1.legend(loc='best', bbox_to_anchor=(1.05, 1.0))
    ax1.set_title('Most optimal curves of each iteration superimposed onto gradient image', fontsize=18)
        
    # Plot costs of the most optimal curves against iteration.
    ax2 = fig.add_subplot(212)
    ax2.scatter(np.arange(1,N_iter+1), iter_optimal_costs, c='r', s=50, edgecolors=(0, 0, 0))
    ax2.set_title('Costs from optimal curves for each iteration', fontsize=18)
    ax2.set_xlabel('Iteration', fontsize=15)
    ax2.set_ylabel('Cost', fontsize=15)
    ax2.set_xticks([i for i in range(1,N_iter+1)])
    
    
    
#================================= DEDUCE WHERE BEST CURVES INTERSECTED GRADIENT IMAGE =================================

def idx_visited(best_curves, size, costs):
    '''
    This function updates the array of visited squares in the gridded search region specified by the bounds and the
    stepsizes delta_x and delta_y
    
    INPUTS:
    ----------
        best_curves (3d-Array) : Best curves from iteration stored in an array of shape: (N, N_samples*N_keep, 2)

        size (tuple) : Size of image
        
        costs (array) : Line integral costs of each of the best curves
    '''

    # Extract size of image
    M,N = size
    N_keep = best_curves.shape[1]

    # If the visisted array isn't specified, initialise a zero array of size (H,W)
    visited_arr = np.zeros((M,N))

    # Given the inputted point, compute the bottom-left vertex of the square that the point exists in  
    i = np.rint(best_curves[:,:,1]).astype('int')
    j = np.rint(best_curves[:,:,0]).astype('int')
    
    # Compute weights for each curve. These are based on a shared voting system where there are only 100
    # votes overall and they are chosen based on the inverse cost (higher cost => better curve) proportions
    # for all N_keep curves. This means the best curve of the group gives their pixel locations a higher
    # weighting.
    inv_costs = 1/costs
    weights = (inv_costs / np.sum(inv_costs)) * N_keep

    # Update the array index corresponding to the vertex as being visited. Weight the score that
    # each pixel location gets by the rank of each curve. This means that pixel locations
    # which are intersected by top-ranked curves score higher in each iteration.
    for idx in range(N_keep):
        visited_arr[i[:, idx],j[:, idx]] += weights[idx]
        
    #Normalise the intersection scores between 0 and 1.
    visited_arr = visited_arr/visited_arr.max()
        
    return visited_arr



#================================= COMPUTE PIXEL SCORE PER ACCEPTED POINT ================================

def comp_pixel_score(pixel_idx, grad_img, visited_arr, score_thresh, N_keep, pre_fobs_flag=False):
    '''
    This function computes the scores for all the pixels which have had a non-zero count of curve intersections, 
    and then only returns those pixels whose score is above the threshold given by score_thresh, in xy-space. 
    
    INPUTS:
    ---------
        pixel_idx (array) : Array of pixel indexes which had a non-zero count of curve intersections from best 
        curves.

        grad_img (array) : Gradient of image.

        visited_arr (array) : Array the same size of the image, storing each pixel indexes curve intersection 
        count.

        score_thresh (float) : Float in [0,1] which decides on the threshold of accepting or rejecting the fixed 
        points.

        N_keep (int)  : Number of samples.

        pre_fobs_flag (bool) : Flag to see if observations being scores are from the previous iteration or from 
        the new iteration.
    '''
    # Compute their gradient value and normalized intersection score of the best curves
    grad_vals = grad_img[pixel_idx[:,0], pixel_idx[:,1]]
    intersection_vals = visited_arr[pixel_idx[:,0], pixel_idx[:,1]]
    
    # If computing new scores for previous observations, remove those which have no intersections with best curves of new
    # iteration
    if pre_fobs_flag:
        intersection_idx = np.logical_not(intersection_vals==0)
        pixel_idx = pixel_idx[intersection_idx]
        grad_vals = grad_vals[intersection_idx]
        intersection_vals = intersection_vals[intersection_idx]

    # Compute score for each pixel based on gradient value and # of curve intersections. Concatenate best pixels and scores by 
    # thresholding scores. Ensure to switch columns of pixels to turn them from pixel-space (yx-space) to xy-space.
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

        delta_x (int) : Stepsize to discretize the x-axis from [0, N] creating N//delta_x 
        subintervals to bin the fixed points.

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
    fobs_scores = np.asarray(fobs_scores).reshape(-1,3)
    fobs = fobs_scores[:,:2].astype(int)
    
    return fobs, fobs_scores



#================================= COMPUTE ACCEPTED FIXED POINTS PER ITERATION =================================

def comp_best_pixels(best_curves, grad_img, score_thresh, delta_x, size, init, costs):
    '''
    This function determines the next set of fixed points to fit to the Gaussian Process
    
    INPUTS:
    ---------
        best_curves (array) :  Best curves from iteration stored in an array of shape: (N, N_samples, 2)

        grad_img (array) : Gradient of image

        score_thresh (float) : Float in [0,1] which decides on the threshold of accepting or rejecting the fixed 
        points

        delta_x (int) : Stepsize to discretize the x-axis from [0, N] creating N//delta_x subintervals to bin the 
        fixed points.
        
        size (tuple) : Size of image
        
        init (array) : Array containing the initial start and end points of edge in xy-space.
        
        costs (array) : Line integral costs of each of the best curves
    '''
    # Determine where in the image the best curves have visited
    M,N = size
    N_keep = best_curves.shape[1]
    visited_arr = idx_visited(best_curves, size, costs)
    x_st = init[0,0]
    x_en = init[1,0]
    
    # Remove first and last columns since all egdes are pinned down to these points.
    pixel_idx = np.argwhere(visited_arr > 0)
    pixel_idx = pixel_idx[(pixel_idx[:,1] != x_st) & (pixel_idx[:,1] != x_en)]
    
    # Compute pixel score and threshold pixels using score_thresh
    best_pts_scores = comp_pixel_score(pixel_idx, grad_img, visited_arr, score_thresh, N_keep)

    # Bin best points into sub-intervals of length delta_x and choose highest scoring point per sub-interval
    fobs, fobs_scores = bin_pts(N, delta_x, best_pts_scores)
    
    return fobs, fobs_scores, visited_arr



#================================= COMPUTE ITERATION OF EDGE TRACING ALGORITHM =================================

def compute_iter(init, y_samples, x_grid, keep_ratio, score_thresh, delta_x, grad_interp, 
                 grad_img, pre_fobs, size, noise_y):
    '''
    This computes an iteration of the edge tracing algorithm. It will compute the cost of all the curves sampled 
    from the gaussian process of the previous iteration. Using only the best keep_ratio of them, a selection 
    of fixed points are accepted to be fitted to the gaussian process model through thresholding using 
    score_thresh, based on each potential fixed points gradient value and a weighting of 'best' curves passing through 
    said fixed points. 
    
    There is also an option reduce the value that is added to the diagonal of the covariance matrix of the 
    gaussian proces to decrease uncertainty (noise) to the observations so that samples from the posterior 
    gaussian process do not always pass through these fixed points.
    
    INPUTS:
    ---------
        init (array) : Array containing the start and last indexes of true edge. Shape: (2,2)

        y_samples (array) : Array of curves sampled from gaussian process of previous iteration. 
        Shape: (N, N_samples)
        
        x_grid (array) : Vector storing the discretized x-axis in the interval [0, N]. Shape: (N, 1)

        keep_ratio (float) : Float between 0 and 1 to denote how many of the best curves to use to determine the 
        fixed points for this iteration.

        score_thresh (float) : float between 0 and 1 to denote the threshold between accepting and rejecting fixed 
        points based on their score from their gradient image reponse and number of best curve intersection scores.

        delta_x (int) : Integer to denote break up the interval [0,N] into N//delta_x sub-intervals to avoid 
        fitting fixed points that are too close to one another. If there are multiple fixed points accepted for a 
        particular sub-interval then only one point is chosen whose score is the highest.

        grad_interp (func) : Gradient interpolation function for computing costs on the curves.

        grad_img (array) : Gradient image. Shape: (M,N)

        pre_fobs (array) : Array of accepted fixed points from previous iteration. Shape: (N_fixed_pts, 2).
        
        size (tuple) : Size of image.
        
        noise_y (float) : Initial noise variance of new observations.
    '''
    
    # Extract image size, determine number of samples and compute number of sub-intervals.
    M,N = size
    N_samples = y_samples.shape[1]
    N_subintervals = int(N//delta_x)
    
    # Compute best curves via cost function and store optimal curve
    best_curves, best_costs, (optimal_curve, optimal_cost) = get_best_curves(grad_interp, y_samples, x_grid, keep_ratio)

    # Determine the set of fixed points which exceed the score threshold for the current iteration. 
    new_fobs, new_fobs_scores, visited_arr = comp_best_pixels(best_curves, grad_img, score_thresh, 
                                                             delta_x, size, init, best_costs)
    N_new_fobs = new_fobs.shape[0]
    
    # Determine if there are old observations to be re-scored or not using pre_fobs_flag
    N_pre_fobs = pre_fobs.shape[0]
    flags = [False, True]
    pre_fobs_flag = flags[np.argmax([0, N_pre_fobs])]
    
    # If the new set of observations is smaller than the previous set (or is too small for first iteration) reduce score and
    # try find more observations.
    while N_new_fobs - N_pre_fobs < 2:
        print('Not enough observations found. Reducing score threshold to {}'.format(round(0.95*score_thresh, 4)))
        score_thresh = round(0.95*score_thresh, 4)
        new_fobs, new_fobs_scores, visited_arr = comp_best_pixels(best_curves, grad_img, score_thresh, 
                                                                  delta_x, size, init, best_costs)
        N_new_fobs = new_fobs.shape[0]
    
    # If there are previous observations, recompute the scores for the previous observations and discard those with 
    # no intersection score.
    # Combine the old and new set of observations and bin them accordingly so only one observation is chosen for each 
    # sub-interval.
    all_fobs = new_fobs
    if pre_fobs_flag:
        pre_fobs_scores = comp_pixel_score(pre_fobs[:, [1,0]], grad_img, visited_arr, score_thresh, N_samples, pre_fobs_flag) 
        combine_fobs = np.concatenate((pre_fobs_scores, new_fobs_scores), axis=0)
        all_fobs, _ = bin_pts(N, delta_x, combine_fobs)

    # If there are observations covering 90% of the sub-intervals then reduce noise observation of pixels. This has the effect
    # of sampling curves from the Gaussian process which lie closer to the observations added to the model
    if all_fobs.shape[0] >= 0.9*N_subintervals and (N_pre_fobs < 0.9*N_subintervals):
        print('90% of subintervals observed. Reducing observation noise to {}'.format(noise_y/10))
        noise_y /= 10
        
    return all_fobs, noise_y, score_thresh, (optimal_curve, optimal_cost)



#================================= RUN ENTIRE ALGORITHM TO OUTPUT EDGE =================================

def run_algorithm(init, grad_img, kernel_params, obs=np.array([]), noise_y=100, N_samples=1000, score_thresh=0.5, delta_x=10, 
                  keep_ratio=0.1, print_diagnostics=True, seed=1522100, init_noise_y=1e-7, incl_final=True):
    '''
    Outer function that runs the Gaussian process edge tracing algorithm. The inputs are chosen by the user
    
    INPUTS:
    -----------
        init (2D-array) : Array of start and end indexes of edge, i.e. init= np.array([[st_x, st_y],[en_x, en_y]]).
        Must be inputted by user.

        grad_img (array) : Gradient of image. Must be inputted by user.
        
        kernel_params (dict) : Dictionary of parameters for the kernel in the Gaussian Process model. Must be inputted by user
                               Will require inputs: 'kernel':'Matern', 'RBF', 'Polynomial',
                                                    'sigma_f', 'sigma_l' and any kernel-specific parameter.
                                                    
        obs (2d-array) : Observations to fit to the model. Usually used when propagating points from a different edge to 
        improve convergence of the current edge of interest. Default value: np.array([]) (empty array).

        noise_y (float) : Amount of noise to perturb the samples drawn from Gaussian process from the non-endpoint observations.
        Default value: 100.
    
        N_samples (int) : Number of samples to draw from Gaussian process. Default value: 1000.
        
        score_thresh (float) : Initial pixel score to determine which observations to accept and fit to the Gaussian process in
        each iteration. Default value: 0.5.
        
        delta_x  (int) : Length of sub-intervals to split x-axis into. Default value: 10. Number of subintervals=N//delta_x 
        where N is the width of the image.
        
        keep_ratio (float) : Proportion of best curves to use to score and choose observations. Default value: 0.1.
                                                    
        print_diagnostics (bool) : Flag to plot the optimal curves and their costs on the gradient image once algorithm
        converges. Default value: False.
        
        seed (int) : Seed to fix random sampling of Gaussian Process. Default value: 1522100.
        
        init_noise_y (float) : Observation noise to pertub end points of edge - typically altered when using propagation from a
        previous edge to the current edge of interest. Default value: 1e-7.
        
        incl_final (bool) : Flag to include the cost of the optimal curve of the final iteration when selecting the outputting edge.
        Default value: True.
    '''
    # Extract size of image, compute interpolated gradient intensity function and compute the x-component pixel indexes for
    # the length of the edge dictated by init
    size = grad_img.shape
    M, N = size
    x_grid = np.arange(N).astype(int)
    
    # Plot random samples drawn from Gaussian process with chosen kernel and ask to continue with algorithm
    #_ = fit_predict_GP(kernel_params, init_noise_y, init, N_samples, np.array([]), 20, True, True, size)
    _ = fit_predict_GP(kernel_params, noise_y, init, size, N_samples, obs, seed, 
                       plt_prior_post=True, N_plt_samples=20, init_noise_y=init_noise_y)
    print('Are you happy with your choice of kernel? Answer with "Yes" or "No"')
    cont = input()
    if cont.lower() != 'yes':
        return
    
    # Compute interplated gradient surface
    grad_interp = grad_interpolation(grad_img, size)
    
    # Initialise the number of sub-intervals dividing the x-axis, the number of observations added to the Gaussian process
    # and the two lists storing the optimal curve and its cost in each iteration
    N_subintervals = int(N//delta_x)
    pre_fobs = obs
    n_fobs = pre_fobs.shape[0]
    iter_optimal_curves = []
    iter_optimal_costs = []
    
    # Convergence of the algorithm is when there is an observation fitted for each sub-interval
    N_iter = 1
    while n_fobs < N_subintervals-1:
        # Start timer for iteration
        st = t.time()
        
        # Intiialise Gaussian process by fitting initial points, outputting samples drawn from initial posterior distribution 
        print('Fitting Gaussian process...')
        _, y_samples = fit_predict_GP(kernel_params, noise_y, init, size, N_samples, pre_fobs, seed,
                                      False, 0, init_noise_y)

        # Compute a single iteration of the algorithm to output new set of observations, new values of noise_y and score_thresh
        # as well as the optimal curve and its cost to be appended to separate lists for plotting at the end of the algorithm.
        print('Computing next set of observations...')
        output = compute_iter(init, y_samples, x_grid, keep_ratio, score_thresh, delta_x, 
                              grad_interp, grad_img, pre_fobs, size, noise_y)
        pre_fobs, noise_y, score_thresh, (optimal_curve, optimal_cost) = output
        iter_optimal_curves.append(optimal_curve)
        iter_optimal_costs.append(optimal_cost)

        # Recompute number of observations
        n_fobs = pre_fobs.shape[0]
        #print(pre_fobs)
        print('Number of observations: {}'.format(n_fobs))

        # Incremement number of iterations and output time taken to perform iteration
        en = t.time()
        print('Iteration {} - Time Elapsed: {}\n\n'.format(N_iter, round(en-st, 4)))
        N_iter += 1
        
    # When convergence is reached, trial several different values of alpha and choose alpha which produces highest scoring 
    # mean curve. Output this optimal mean curve in pixel space as the trace of the edge.
    alpha_curves = []
    alpha_costs = []
    alpha_final = []
    potential_alpha = [1, 0.1, 0.01, 0.001, 0.0001, 1e-5]
    for alpha in potential_alpha:
        try:
            y_mean, _ = fit_predict_GP(kernel_params, alpha, init, size, N_samples, pre_fobs, seed,
                                      False, 0, init_noise_y)
            mean_curve = np.concatenate((x_grid[:,np.newaxis], y_mean[:, np.newaxis]), axis=1)
            alpha_curves.append(mean_curve)
            alpha_costs.append(cost_funct(grad_interp, mean_curve))  
            alpha_final.append(alpha)
        except:
            pass
        
    if incl_final:
        alpha_curves.append(iter_optimal_curves[-1])
        alpha_costs.append(iter_optimal_costs[-1])
        alpha_final.append(noise_y)
        
    edge_trace = alpha_curves[np.argmin(alpha_costs)][:, [1,0]].astype(int)
    
    # When convergence is reached, option to plot the optimal curves on the gradient image and their costs on a separate plot.
    if print_diagnostics:
        print('Final noise observation selected: {}'.format(alpha_final[np.argmin(alpha_costs)]))
        iter_optimal_curves.append(edge_trace[:, [1,0]])
        iter_optimal_costs.append(min(alpha_costs))
        plot_diagnostics(grad_img, iter_optimal_curves, iter_optimal_costs, size)
    
    return edge_trace
