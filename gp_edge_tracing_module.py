# -*- coding: utf-8 -*-
"""
Created on Wed 19 May 15:06:44 2021

@author: s1522100

This module traces an edge in an image using Gaussian process regression.

"""
import numpy as np
import os
import numpy.matlib as npml
import time as t
import sklearn.gaussian_process as skgp
import matplotlib.pyplot as plt

from gpet_gpr import GaussianProcessRegressor as GPR
from gpet_gpr import WeightedWhiteKernel
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import simps
from KDEpy import FFTKDE

class GP_Edge_Tracing(object):
    '''
    This module traces an individual edge in an image using Gaussian process regression.
    '''
    
    def __init__(self, init, grad_img, kernel_options=(1,3,3), noise_y=1, obs=np.array([]), N_samples=1000,
                score_thresh=0.5, delta_x=10, keep_ratio=0.1, seed=1, bandwidth=1, return_std=False, fix_endpoints=False):
        '''
        Internal parameters of the edge tracing algorithm.
        
        INPUTS:
        -----------
            init (2D-array) : Array of start and end indexes of edge, i.e. init= np.array([[st_x, st_y],[en_x, en_y]]).
            Must be inputted by user.

            grad_img (array) : Gradient of image. Must be inputted by user. Assumed to be normalised between [0, 1].

            kernel_options (tuple or dict) : For ease in automation, this provides the user with the option of which kernel to 
            go for. Will be a tuple of integers with 2 options for kernel type (RBF or Matern), 5 options for the function variance 
            and 5 options for the lengthscale which are scaled according to the image height and length of edge.
            
            noise_y (float) : Amount of noise to perturb the samples drawn from Gaussian process from the non-endpoint observations.
            Default value: 1.

            obs (2d-array) : Observations to fit to the model. Usually used when propagating points from a different edge to 
            improve convergence of the current edge of interest. Default value: np.array([]) (empty array).

            N_samples (int) : Number of samples to draw from Gaussian process. Default value: 1000.

            score_thresh (float) : Initial pixel score to determine which observations to accept and fit to the Gaussian process in
            each iteration. Default value: 0.5.

            delta_x  (int) : Length of sub-intervals to split x-axis into. Default value: 10. Number of subintervals=N//delta_x 
            where N is the width of the image.

            keep_ratio (float) : Proportion of best curves to use to score and choose observations. Default value: 0.1.

            seed (int) : Seed to fix random sampling of Gaussian Process. Default value: 1522100.     
            
            bandwidth (float) : Bandwith defining the radius of the Gaussian kernel when computing weighted density function
            of the optimal posterior curves.
            
            return_std (bool) : If flagged, return 95% credible interval as well as edge trace.
            
            fix_endpoints (bool) : If flagged, user-inputted edge endpoints are fixed (negligible observation noise is chosen for
            these pixels).
        '''
        
        # Set global internal parameters
        self.init = init
        self.x_st, self.x_en = int(init[0,0]), int(init[-1,0])
        self.grad_img = grad_img
        self.noise_y = 100 * noise_y
        self.kernel_options = kernel_options
        self.N_samples = N_samples
        self.obs = obs
        self.seed = seed
        self.bandwidth = bandwidth
        self.keep_ratio = keep_ratio
        self.score_thresh = score_thresh
        self.delta_x = int(delta_x) if delta_x > 3 else 2
        self.return_std = return_std
        self.fix_endpoints = fix_endpoints
        
        # Set up gloabl internal parameters using inputs.
        self.size = grad_img.shape
        self.M, self.N = self.size
        self.x_grid = self.x_st + np.arange(self.x_en - self.x_st + 1).astype(int)
        self.edge_length = self.x_grid.shape[0]
        self.N_subints = int(self.edge_length // self.delta_x)
        self.N_keep = int(keep_ratio*N_samples)
        
        # Compute interpolated gradient surface and compute KDE of image gradient
        self.grad_kde = self.kernel_density_estimate(grad_img=True)
        self.grad_interp = self.grad_interpolation(grad_img)
        
        # Set up check for kernel, function variance and length scale
        if type(self.kernel_options) == dict:
            self.sigma_f = kernel_options['sigma_f']
            self.sigma_l = kernel_options['length_scale']
            self.kernel_type = kernel_options['kernel']
            self.kernel_nu = kernel_options['nu'] if kernel_options['kernel']=='Matern' else 2.5
        else:
            rbf_matern, sigmaf_opt, sigmal_opt = kernel_options

            #rbf or matern
            if rbf_matern == 0:
                self.kernel_type = 'RBF'
            else:
                self.kernel_type = 'Matern'
                if rbf_matern == 1:
                    self.kernel_nu = 2.5
                else:
                    self.kernel_nu = 1.5

            # function variance
            if sigmaf_opt == 1:
                self.sigma_f = self.M // 10
            elif sigmaf_opt == 2:
                self.sigma_f = self.M // 8
            elif sigmaf_opt == 3:
                self.sigma_f = self.M // 6
            elif sigmaf_opt == 4:
                self.sigma_f = self.M // 4
            else:
                self.sigma_f = self.M // 2

            # length scale
            if sigmal_opt == 1:
                self.sigma_l = self.x_grid.shape[0]
            elif sigmal_opt == 2:
                self.sigma_l = (3*self.x_grid.shape[0])//4
            elif sigmal_opt == 3:
                self.sigma_l = self.x_grid.shape[0]//2
            elif sigmal_opt == 4:
                self.sigma_l = self.x_grid.shape[0]//4
            else:
                self.sigma_l = self.x_grid.shape[0]//10        
       
    
    def fit_predict_GP(self, obs, plt_post=False, N_plt_samples=20, converged=False, seed=1, 
                       true_edge=None, credible_interval=False):
        ''' 
        Fits a Gaussian Process using init and obs. The coordinates in obs are perturbed using observation noise, noise_y.
        N_samples posterior curves are sampled and outputted (and can be fixed using seed). Function also has option to plot
        20 sample curves, mean function, 95% confidence interval and observatons/init/noisy_pts too.
            
        INPUTS:
        ---------------
            plt_post (bool) : Flag to plot posterior distribution using N_plt_samples curves sampled from 
            Gaussian process. Default value: False.

            N_plt_samples (int) : Number of curves to sample from Gaussian process to sample if plt_post=True. 
            Default value: 20.

            converged (bool) : If each sub-interval has an accepted pixel coordinate, then converged=True and only need to
            draw mean function. Otherwise, only draw sample curves.
            
            seed (int) : Seed to fix random posterior curve sampling.
            
            true_edge (array) : If true edge is known, this will be plotted if plt_plot=True
            
            credible_interval (bool) : If flagged, final GPR is retrained with AND without optimising observation noise so that
            95% credible intervals can be outputted.
        '''

        # Set up elements of kernel and optimisation parameters for GPR
        gp_params = {'normalize_y':True, 'alpha':1e-10, 'copy_X_train':True, 'random_state':seed} 
        gp_params['optimizer'] = 'fmin_l_bfgs_b'
        gp_params['n_restarts_optimizer'] = 5
        if converged:
            gp_params['optimizer'] = 'fmin_l_bfgs_b'
            gp_params['n_restarts_optimizer'] = 5
            if credible_interval:
                noise_bounds = 'fixed'
            else:
                noise_bounds = (1e-7, self.noise_y)
            const_bounds = (1e-3*self.sigma_f, 1e3*self.sigma_f)
            length_bounds = (1e-3*self.sigma_l, 1e3*self.sigma_l)
        else:
            gp_params['optimizer'] = None
            gp_params['n_restarts_optimizer'] = 0
            const_bounds = 'fixed'
            noise_bounds = 'fixed'
            length_bounds = 'fixed'
            
        # Set up kernel parameters for weighted white noise kernel
        # Testing out placing noise on endpoints.
        if self.fix_endpoints:
            alpha_init = 1e-7*np.ones((self.init.shape[0]))
        else:
            alpha_init = 0.5*np.ones((self.init.shape[0]))
        alpha_obs = np.ones((obs.shape[0]))
        alpha = np.concatenate([alpha_init, alpha_obs], axis=0)
        
        # Set up constant kernel and weighted white noise kernel 
        white_kernel = WeightedWhiteKernel(edge_length=self.x_grid.shape[0], noise_weight=alpha, 
                                      noise_level=self.noise_y, noise_level_bounds=noise_bounds)
        constant_kernel = skgp.kernels.ConstantKernel(self.sigma_f**2, constant_value_bounds=const_bounds)
        
        # Define kernel for GP
        if self.kernel_type == 'Matern':
            kernel = constant_kernel * skgp.kernels.Matern(nu = self.kernel_nu, length_scale=self.sigma_l, 
                                                           length_scale_bounds=length_bounds) + white_kernel
        else:
            kernel = constant_kernel * skgp.kernels.RBF(length_scale=self.sigma_l, length_scale_bounds=length_bounds) + white_kernel

        # Construct Gaussian process regressor
        gp_params['kernel'] = kernel
        gp = GPR(**gp_params)
        
        # Construct inputs and outputs
        obs = obs.reshape(-1,2)
        new_obs = np.concatenate([self.init, obs], axis=0)
        X = new_obs[:,0][:, np.newaxis]
        y = new_obs[:,1]
    
        # Fit data to the GP to update mean and covariance matrix
        gp.fit(X, y)
        
        # Sample posterior curves, output posterior predictive mean function and standard deviations
        # Extract both samples, mean function and standard deviation for plotting
        y_samples = gp.sample_y(self.x_grid[:, np.newaxis], self.N_samples, random_state=seed)
        y_mean, y_std = gp.predict(self.x_grid[:, np.newaxis], return_std=True)
            
        # For plotting
        if plt_post:
            y_plt_samples = y_samples[:,:N_plt_samples]
            fontsize = 20
            fig, ax = plt.subplots(figsize=(15,10))

            # Plot mean function, 95% confidence interval and N_plt_samples curves
            plt_mean = ax.plot(self.x_grid, y_mean, 'k', lw=3, zorder=3, label='Posterior Predictive Mean')
            plt_sd = ax.fill_between(self.x_grid, y_mean - 2*y_std, y_mean + 2*y_std, 
                                       alpha=0.2, color='k', zorder=1, label='95% Credible Region')
            plt_samples = ax.plot(self.x_grid, y_plt_samples, lw=1, zorder=2)

            # Overlay initial endpoints and observations
            plt_init = ax.scatter(self.init[:, 0], self.init[:,1], c='m', s=5*fontsize, zorder=4,
                                        edgecolors=(0, 0, 0), label='Edge Endpoints')
            if obs.size > 0:
                post_obs = ax.scatter(obs[:,0], obs[:,1], c='r', s=3*fontsize, 
                                       zorder=4, edgecolors=(0, 0, 0), label='Observations')

            # If converged and optimising observation noise, then superimpose optimal kernel. 
            # Otherwise, just set title as chosen kernel
            if not converged: # Added kernel_title here 02/09/2021
                if self.kernel_type == 'RBF':
                    kernel_title = (f'{self.sigma_f}**2 * {self.kernel_type}(length_scale={self.sigma_l})'
                                    f' + WeightedWhiteKernel(noise_level={self.noise_y/100})')
                elif self.kernel_type == 'Matern':
                    kernel_title = (f'{self.sigma_f}**2 * {self.kernel_type}(length_scale={self.sigma_l}, nu={self.kernel_nu})'
                                    f' + WeightedWhiteKernel(noise_level={self.noise_y/100})')
                    
                ax.set_title(f"Posterior Distribution\n Kernel: {kernel_title}.",fontsize=fontsize)
            else:
                ax.set_title(f"Posterior Distribution\n Optimal Kernel: {gp.kernel_}.",fontsize=fontsize)
                
            # If true_edge is provided, plot that too
            if true_edge is not None:
                tr_edge = ax.plot(true_edge[:,1], true_edge[:,0], 'r', lw=3, zorder=3, label='True Edge')
                
            ax.set_xlim([0, self.N-1])
            ax.set_ylim([self.M-1, 0])
            ax.set_xlabel('Pixel Column, $x$', fontsize=fontsize)
            ax.set_ylabel('Pixel Row, $y$', fontsize=fontsize)            
            #ax.set_yticks(np.array(ax.get_yticks()).astype(int))
            #ax.set_xticks(np.array(ax.get_xticks()).astype(int))
            #ax.set_xticklabels(ax.get_yticks().tolist(), fontsize=fontsize-5)
            #ax.set_yticklabels(ax.get_yticks().tolist(), fontsize=fontsize-5)
            handles, labels = ax.get_legend_handles_labels()
            ax_legend = ax.legend(handles, labels, fontsize=25, ncol=2, 
                      loc='lower right', edgecolor=(0, 0, 0, 1.))
            ax_legend.get_frame().set_alpha(None)
            ax_legend.get_frame().set_facecolor((1,1,1,1))
            fig.tight_layout()
            plt.show()
            plt.pause(0.5)
        
        if not converged:
            return y_mean, y_samples#, fig, y_std #for showing algorithm as particular iterations
        elif converged and not credible_interval:
            return y_mean
        elif converged and credible_interval:
            return y_mean, y_std
    
    
    def grad_interpolation(self, grad_img, gmin=1e-12):
        '''
        Linearly interpolates the image gradient to evaluate at real-valued coordinates.

        INPUTS:
        -----------
            grad_img (2D-array) : Gradient image.

            gmin (float) : Values to fill 0-values in gradient image.
        '''

        # Create meshgrid of integer points denoting indices of image
        x = np.arange(self.N)
        y = np.arange(self.M)

        # Create meshgrid
        X, Y = np.meshgrid(x, y)
        XY = np.stack([X.ravel(), Y.ravel()]).T

        # The surface here is the gradient image. Mark any 0-values to gmin
        grad_img_cpy = self.grad_img.copy()
        grad_img_cpy[grad_img_cpy == 0] = gmin
        Z = grad_img_cpy.ravel()

        # Perform basic linear 2-dimensional interpolation to create gradient surface. Ensure that if evaluating
        # at point outside input domain, set value to 0
        grad_interp = LinearNDInterpolator(XY, Z, fill_value = 0)

        return grad_interp 

    
    
    def finite_diff(self, y, h=1, typ='forward'):
        '''
        Function to approximate derivative of function using finite difference methods. A choice of central, forward
        and backward methods are available.

        INPUTS:
        ---------------------
            y (array-like) : Array of function values sampled at discrete time points

            h (int) : Stepsize. Assumed to be 1.

            typ (str) : Type of finite difference method. Assumes forward differencing.
        '''
        # Depending on the type of finite difference method employed, the diff list stores changes in consecutive
        # elements of y and returns as an array.
        N = y.shape[0]
        diff = []
        if typ == 'forward':
            for i in range(N-1):
                diff.append(y[i+h] - y[i])
        elif typ == 'backward':
            for i in range(1, N):
                diff.append(y[i] - y[i-h])
        elif typ == 'central':
            for i in range(1, N-1):
                diff.append(y[i-h] - y[i+h])

        return np.array(diff) 
    
    
    
    def cost_funct(self, edge):
        '''
        Computes the line integral per unit pixel length along the specified edge given the image gradient. We numerically 
        integrate the edge by computing the curvilinear coordinates along the image gradient using euclidean distance and 
        Simpson's rule. This is then normalised by the length of the edge to penalise sharp movements.

        INPUTS:
        -----------
            edge (array) : edge indexes in xy-space, shape=(N, 2), i.e. new_edge = np.array([[x0, y0],...[xN, yN]])
        '''
        # Evaluate edge along interpolated gradient image
        edge = edge[edge[:,0].argsort(), :]
        grad_score = self.grad_interp(edge)

        # Compute cumulative sum of euclidean distance between pixel indexes of edge.
        # This is the equivalent of computing the curvilinear coordinates of the edge
        pixel_diff = np.cumsum(np.sqrt(np.sum(np.diff(edge, axis=0)**2, axis=1)))

        # Compute integrand of arc length of edge
        pixel_deriv = self.finite_diff(edge[:,1], h=1, typ='forward')
        integrand = np.sqrt(1 + pixel_deriv**2)

        # Compute cost = 1 / (line integral / arc length) = arc length / line integral - smaller costs are better 
        line_integral = simps(grad_score[:-1], pixel_diff)
        arc_length = simps(integrand, edge[:-1,0])
        
        return arc_length / line_integral 
    
    
    
    def get_best_curves(self, y_samples):
        '''
        This function works out the optimal posterior curves sampled from the Gaussian process posterior distribution 
        of the current iteration. It outputs these best curves, their respective costs and the most optimal curve and 
        cost (for plotting purposes).

        INPUTS:
        ----------
            y_samples (array) : N_samples posterior curves sampled from the gaussian process posterior of the current iteration.
        '''
        # Stack the samples and their input locations to be fetched for computing their cost.
        X = npml.repmat(self.x_grid, self.N_samples, 1).T
        curves = np.stack((X, y_samples), axis=2)

        # Initialise list costs of these best curves and compute costs of each best curve
        costs = []
        for i in range(self.N_samples):
            costs.append(self.cost_funct(curves[:,i,:]))
        costs = np.asarray(costs)

        # Select N_keep curves as the best curves and their costs
        best_idxs = np.argsort(costs)[: self.N_keep]
        best_curves = curves[:, best_idxs, :]
        best_costs = costs[best_idxs]

        # Store optimal curve and its cost
        optimal_curve = best_curves[:,0,:]
        optimal_cost = best_costs[0]

        return best_curves, best_costs, (optimal_curve, optimal_cost)
    
        
     
        
    def kernel_density_estimate(self, best_curves=None, costs=None, normalise=True, grad_img=False):
        '''
        This function estimates the kernel density function of the Gaussian process' posterior distributions belief in where
        the edge of interest lies, through using the finite sample distribution of the most optimal posterior curves. This also
        has an option to estimate kernel density function of the image gradient.

        INPUTS:
        ----------
            best_curves (Array) : Best curves from iteration stored in an array of shape: (N, N_samples*keep_ratio, 2), where
            N is the length of the edge of interest.

            costs (array) : Costs of each of the optimal posterior curves.

            normalise (bool) : Whether to normalise PDF between 0 and 1 rather than leave it as a well-defined PDF (where integrating 
            across the domain equals 1).

            bw (float) : Bandwidth of Gaussian kernel. Taken out on 09/09/2021 to allow user-inputted parameter self.bandwidth

            grad_img (bool) : Boolean value of whether to compute kernel density estimate of image gradient or not.
        '''

        # If image gradient is not inputted, compute KDE of curve intersection
        if not grad_img:
            # Extract sample points from the best curves by reshaping array
            sample_pts = best_curves.reshape(-1,2)
            N_curve = best_curves.shape[0]

            # Compute weights for each sample point. Sample points belonging to the optimal curves get a higher weight 
            # (quantified through the reciprocal of the cost function). 
            inv_costs = 1/costs
            weights = (inv_costs / np.sum(inv_costs))
            weights_arr = np.tile(weights, (N_curve, 1))
            weights_arr = weights_arr.reshape(-1,)
            
            # If any points are outside of the image, remove before estimating kernel density function
            out_of_domain_pts = np.argwhere((sample_pts[:,1] < 0) | (sample_pts[:,1] > self.M-1))
            sample_pts = np.delete(sample_pts, out_of_domain_pts, axis=0)
            weights_arr = np.delete(weights_arr, out_of_domain_pts, axis=0)
            
            # Define bandwidth
            bw = self.bandwidth
        else:
            # Extract non-zero gradient intensity pixel coordinates. Weight each coordinate according
            # to their gradient intensity
            sample_pts = np.argwhere(self.grad_img > 1e-4)
            weights_arr = self.grad_img[sample_pts[:,0], sample_pts[:,1]].reshape(-1)
            sample_pts = sample_pts[:, [1,0]].reshape(-1,2)
            
            # Define bandwidth
            bw = 1

        # Estimate kernel density function using a Gaussian kernel with bandwidth matrix [[1, 0], [0, 1]], i.e. 2x2 Identity matrix. 
        # This is equivalent to centering a Gaussian kernel on each sample point with a circle of radius 1 so there is 
        # no dominant orientation and the smoothing constitues to covering the interior of at most 4 adjacent pixel coordinates.
        gauss_kernel = FFTKDE(kernel='gaussian', bw=bw).fit(sample_pts, weights=weights_arr)
        x_range = np.arange(-1, self.N+1)
        y_range = np.arange(-1, self.M+1)
        mesh_xy = np.meshgrid(x_range, y_range)
        xy_range = np.stack([mesh_xy[0].T.ravel(), mesh_xy[1].T.ravel()]).T

        # Evaluate KDE along the pixel coordinates of the image
        disc_kde = gauss_kernel.evaluate(xy_range).reshape((self.N+2, self.M+2)).T
        disc_kde = disc_kde[1:-1, 1:-1]

        # If normalise, min-max normalise the PDF
        if normalise:
            disc_kde -= disc_kde.min()
            disc_kde /= disc_kde.max()

        return disc_kde   
        
        
        
    def comp_pixel_score(self, pixel_idx, kde_arr, pre_fobs_flag=False):
        '''
        This function computes the scores for all the pixels which have had a large enough density on the kernel density 
        estimate ofrom the optimal posterior curves. If scoring previous observations, then some of these may have a low
        density on the frequency distribution and are hence discarded. Function returns those pixels whose score is above the 
        threshold given by score_thresh, in xy-space.

        INPUTS:
        ---------
            pixel_idx (array) : Array of pixel indexes which had a non-zero density on the KDE of the optimal posterior
            curves.

            kde_arr (array) : KDE of optimal posterior curves.

            pre_fobs_flag (bool) : Flag to see if observations being scores are from the previous iteration or from 
            the new iteration.
        '''
        # Compute their gradient value and normalized intersection score of the best curves
        grad_vals = self.grad_kde[pixel_idx[:,0], pixel_idx[:,1]]
        intersection_vals = kde_arr[pixel_idx[:,0], pixel_idx[:,1]]

        # If computing new scores for previous observations, remove those which have no intersections with best curves of new
        # iteration
        if pre_fobs_flag:
            intersection_idx = np.logical_not(intersection_vals <= 1e-4)
            pixel_idx = pixel_idx[intersection_idx]
            grad_vals = grad_vals[intersection_idx]
            intersection_vals = intersection_vals[intersection_idx]

        # Compute score for each pixel based on KDE's of image gradient and posterior curves. Concatenate best pixels and 
        # scores by thresholding scores. Ensure to switch columns of pixels to turn them from pixel-space (yx-space) to 
        # xy-space.
        pixel_scores = 1/3 * (intersection_vals * grad_vals + intersection_vals + grad_vals) # score_fn 1
#        pixel_scores = 1/2 * (grad_vals + intersection_vals) #score_fn 2
#        pixel_scores = 1/2 * (grad_vals * intersection_vals + intersection_vals) # score_fn 3
#        pixel_scores = 1/2 * (grad_vals * intersection_vals + grad_vals) # score_fn 4
#        pixel_scores = 1/3 * (2*grad_vals + intersection_vals) # score_fn 5
#        pixel_scores = 1/3 * (grad_vals + 2*intersection_vals) # score_fn 6
#        pixel_scores = 1/4 * (2*grad_vals*intersection_vals + intersection_vals + grad_vals) # score_fn 7
#        pixel_scores = 1/2 * (np.exp(1 - 1/grad_vals) + np.exp(1-1/intersection_vals)) # score_fn 8
        
        best_scores = pixel_scores[pixel_scores > self.score_thresh].reshape(-1,1)
        best_pixels = pixel_idx[np.argwhere(pixel_scores > self.score_thresh)].reshape(-1,2)
        best_pts_scores = np.concatenate((best_pixels[:, [1,0]], best_scores), axis=1)

        return best_pts_scores   
        
                
        
    def bin_pts(self, best_pts_scores):
        '''
        This bins the best pixels into sub-intervals splitting up the x-axis and performs non-max suppression to select the
        highest scoring pixel in each subinterval. These pixel coordinates will be fitted to the Gaussian process in xy-space
        for the next iteraton.

        INPUTS:
        ----------
            best_pts_scores (array) : Array of pixel coordinates whose score surpasses the score threshold.
        '''
        # Bin thresholded pixel coordinates into sub-intervals of length delta_x
        x_visited = (self.N_subints+1)*[[]]
        bin_idx = np.floor((best_pts_scores[:,0]-self.x_st)/self.delta_x).astype(int)
        
        # choose highest scoring pixel coordinate per sub-interval
        fobs_scores = []
        for idx, i in enumerate(np.unique(bin_idx)):
            x_visited[i] = best_pts_scores[np.argwhere(bin_idx == i)].reshape(-1,3)
            fobs_scores.append(x_visited[i][np.argmax(x_visited[i][:,2])])

        # Convert to array and return the pixel coordinates, not the scores as these are updated in each iteration.
        fobs_scores = np.asarray(fobs_scores).reshape(-1,3)
        fobs = fobs_scores[:,:2].astype(int)

        return fobs, fobs_scores        
        
         
    
    def comp_best_pixels(self, best_curves, costs):
        '''
        This function determines the next set of fixed points to fit to the Gaussian Process. It will estimate the density 
        function of the optimal posterior curves, score the pixel coordinates which are large enough to be considered 
        a potential candidate, bin and perform non-max suppression to obtain a pixel coordinate for each subinterval.

        INPUTS:
        ---------
            best_curves (array) :  Optimal posterior curves from iteration stored in an array of 
            shape: (N, N_samples*keep_ratio, 2)

            costs (array) : Costs of each of the optimal posterior curves.
        '''
        # Compute the density function representing the frequency distribution of optimal posterior curves exploring the image
        kde_arr = self.kernel_density_estimate(best_curves, costs)

        # Only store those pixel coordinates as candidates if their density is "reasonably" non-zero (it is set as 1e-4)
        # Also remove first and last columns since all egdes are pinned down to these points and will always have highest 
        # score.
        pixel_idx = np.argwhere(kde_arr > 1e-4)
        pixel_idx = pixel_idx[(pixel_idx[:,1] > self.x_st) & (pixel_idx[:,1] < self.x_en)]

        # Compute pixel score and threshold pixels using score_thresh
        best_pts_scores = self.comp_pixel_score(pixel_idx, kde_arr)

        # Bin best points into sub-intervals of length delta_x and choose highest scoring point per sub-interval. This performs
        # the binning and non-max suppression procedure.
        fobs, fobs_scores = self.bin_pts(best_pts_scores)

        return fobs, fobs_scores, kde_arr    
   
    
    
    def plot_diagnostics(self, iter_optimal_curves, iter_optimal_costs, credint=None):
        '''
        Plot the optimal posterior curves on top of the image gradient. Plot a graph of the costs against iteration.

        INPUTS:
        ----------
            iter_optimal_curves (list) : List of optimal curves for each iteration.

            iter_optimal_costs (list) : List of costs of optimal curves for each iteration.
        '''
        # Plot the most optimal curves from each iteration
        N_iter = len(iter_optimal_curves)
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,25))
        ax1.imshow(self.grad_img, cmap='gray')
        for i, curve in enumerate(iter_optimal_curves[:-1]):
            ax1.plot(self.x_grid, curve[:,1], '--', alpha=0.25, label='Iteration {}'.format(i+1))
        ax1.plot(self.x_grid, iter_optimal_curves[-1][:,1], '-', label='Final Edge')
        if credint is not None:
            ax1.fill_between(self.x_grid, credint[0], credint[1], alpha=0.5, 
                         color='m', zorder=1, label='95% Credible Region')
        ax1.legend(loc='best', bbox_to_anchor=(1.05, 1.0))
        ax1.set_title('Most optimal curves of each iteration superimposed onto gradient image', fontsize=18)

        # Plot costs of the most optimal curves against iteration.
        ax2.scatter(np.arange(1, N_iter+1), iter_optimal_costs, c='r', s=50, edgecolors=(0, 0, 0))
        ax2.set_title('Costs from optimal curves for each iteration', fontsize=18)
        ax2.set_xlabel('Iteration', fontsize=15)
        ax2.set_ylabel('Cost', fontsize=15)
        ax2.set_xticks([i for i in range(1, N_iter+1)])
        
        fig.tight_layout()
        plt.show()
        plt.pause(0.5)
    
    
    
    def compute_iter(self, y_samples, pre_fobs, verbose):
        '''
        This computes an iteration of the edge tracing algorithm. It will compute the cost of all the curves sampled 
        from the gaussian process of the previous iteration. Using only the best keep_ratio of posterior curves, a selection 
        of pixel candidates are accepted to be fitted to the gaussian process model. This is done through a scoring
        system based on the density function estimated with the image gradient and optimal posterior curve. These coordinates 
        are then thresholded, binned and non-max suppression is used to select most optimal pixel coordinates of each iteration.

        The function also ensures that at least one pixel coordinate is added per iteration, by reducing the score_thresh by 5%
        and accepting pixel coordinates with a lower score *if* there are none found. 

        INPUTS:
        ---------
            y_samples (array) : Array of posterior curves sampled from gaussian process of previous iteration. 
            Shape: (N, N_samples)

            pre_fobs (array) : Array of accepted fixed points from previous iteration. Shape: (N_fixed_pts, 2).

            verbose (bool) : If flagged, output text updating user on score threshold reduction.
        '''
        # Compute optimal posterior curves via cost function and store most optimal curve and cost for plotting purposes
        best_curves, best_costs, (optimal_curve, optimal_cost) = self.get_best_curves(y_samples)

        # Determine the set of pixel coordinates which exceed the score threshold for the current iteration. 
        new_fobs, new_fobs_scores, kde_arr = self.comp_best_pixels(best_curves, best_costs)
        N_new_fobs = new_fobs.shape[0]

        # Determine if there are old pixel coordinates to be re-scored or not using pre_fobs_flag
        N_pre_fobs = pre_fobs.shape[0]
        flags = [False, True]
        pre_fobs_flag = flags[np.argmax([0, N_pre_fobs])]

        # If the new set of pixel coordinates is smaller than the previous set (or is too small for first iteration) reduce 
        # score and try find more pixel coordinates with lower score.
        while N_new_fobs - N_pre_fobs < 2:
            # If verbose, print out reduction of score threshold 
            if verbose:
                print(f'Not enough observations found. Reducing score threshold to {round(0.95*self.score_thresh, 9)}')
            self.score_thresh = round(0.95*self.score_thresh, 9)
            new_fobs, new_fobs_scores, kde_arr = self.comp_best_pixels(best_curves, best_costs)
            N_new_fobs = new_fobs.shape[0]

        # If there are previous observations, recompute the scores for the previous observations and discard those with 
        # small density on the frequency distribution score.
        # Combine the old and new set of observations by binning and using non-max suppression so only one pixel coordinate is 
        # chosen for each sub-interval.
        all_fobs = new_fobs
        if pre_fobs_flag:
            pre_fobs_scores = self.comp_pixel_score(pre_fobs[:, [1,0]], kde_arr, pre_fobs_flag) 
            combine_fobs = np.concatenate((pre_fobs_scores, new_fobs_scores), axis=0)
            all_fobs, _ = self.bin_pts(combine_fobs)

        return all_fobs, (optimal_curve, optimal_cost)    
    
    
    
    def gpet(self, print_final_diagnostics=False, show_init_post=False, show_post_iter=False, verbose=False):
        '''
        Outer function that runs the Gaussian process edge tracing algorithm. Pseudocode above describes algorithms procedure.

        INPUTS:
        -----------
            show_init_post (bool) : Flag to show the initial posterior predictive distribution and ask user to continue or not.
            
            show_post_iter (bool) : Flag to show posterior predictive distribution during the fitting procedure.

            print_final_diagnostics (bool) : Flag to plot the optimal curves and their costs on the gradient image once algorithm
            converges. Default value: False.

            verbose (bool) : If flagged, output text updating user on fitting procedure.
        '''
        # Plot random samples drawn from Gaussian process with chosen kernel and ask to continue with algorithm.   
        if show_init_post:
            _ = self.fit_predict_GP(self.obs, plt_post=True, N_plt_samples=20, converged=False)
            print('Are you happy with your choice of kernel? Answer with "Yes" or "No"')
            cont = input()
            if cont.lower() != 'yes':
                return

        # Measure time elapsed
        alg_st = t.time()
        
        # Initialise the number of sub-intervals dividing the x-axis, the number of observations added to the Gaussian process
        # and the two lists storing the optimal curve and its cost in each iteration
        pre_fobs = self.obs
        n_fobs = pre_fobs.shape[0]
        iter_optimal_curves = []
        iter_optimal_costs = []

        # Convergence of the algorithm is when there is an observation fitted for each sub-interval. In this implementation,
        # due to delta_x, I've set the While loop to stop when there is an observation fitted for all sub-intervals bar one.
        N_iter = 1
        while n_fobs < self.N_subints-1:
            # Start timer for iteration
            st = t.time()

            # If verbose, let user know that GP is being fit and new observations are being searched for
            if verbose:
                print('Fitting Gaussian process and computing next set of observations...')

            # Intiialise Gaussian process by fitting initial points, outputting samples drawn from initial posterior 
            # distribution 
            _, y_samples = self.fit_predict_GP(pre_fobs, show_post_iter, 20, converged=False, seed=self.seed+N_iter)

            # Compute a single iteration of the algorithm to output new set of pixel coordinates, new values of ob_noise and 
            # score_thresh as well as the optimal curve and its cost to be appended to separate lists for plotting at the 
            # end of the algorithm.
            pre_fobs, (optimal_curve, optimal_cost) = self.compute_iter(y_samples, pre_fobs, verbose)
            iter_optimal_curves.append(optimal_curve)
            iter_optimal_costs.append(optimal_cost)

            # Recompute number of observations
            n_fobs = pre_fobs.shape[0]

            # Incremement number of iterations and output time taken to perform iteration
            en = t.time()
            N_iter += 1

            # If verbose, print out number of observations 
            if verbose:
                print(f'Number of observations: {n_fobs}')
                print(f'Iteration {N_iter} - Time Elapsed: {round(en-st, 4)}\n\n')

        # To obtain 95% credible interval at termination of algorithm
        if self.return_std:
            y_mean_credint, y_std = self.fit_predict_GP(pre_fobs, print_final_diagnostics, 20, converged=True, 
                                            seed=self.seed+N_iter, credible_interval=True)
            cred_interval = (y_mean_credint - 2*y_std, y_mean_credint + 2*y_std)
        else:
            cred_interval = None
        
        # This will optimise the observation noise and kernel hyperparameters by maximising the marginal likelihood 
        # of the training set of pixel coordinates. This doesn't take into account the optimisation of these pixel 
        # coordinates being edge coordinates of the edge of interest however. Future work would include this. 
        y_mean_optim = self.fit_predict_GP(pre_fobs, print_final_diagnostics, 20, converged=True, 
                                        seed=self.seed+N_iter, credible_interval=False)
        optim_mean_curve = np.concatenate([self.x_grid[:,np.newaxis], y_mean_optim[:, np.newaxis]], axis=1)
        edge_trace = np.rint(optim_mean_curve[:, [1,0]]).astype(int)

        if print_final_diagnostics:
            iter_optimal_curves.append(edge_trace[:, [1,0]])
            iter_optimal_costs.append(self.cost_funct(optim_mean_curve))
            self.plot_diagnostics(iter_optimal_curves, iter_optimal_costs, cred_interval)

        # Compute and print time elapsed
        alg_en = t.time()
#        print(f'Time elapsed before algorithm converged: {round(alg_en-alg_st, 3)}') # Commented out verbosity of algorithm

        if self.return_std:
            return edge_trace, cred_interval
        else:
            return edge_trace
       
