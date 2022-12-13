"""
@author: Jamie Burke
@email: Jamie.Burke@ed.ac.uk

The module provides a framework to trace individual edges in an image using Gaussian process regression.
"""

import numpy as np
import time as t
import sklearn.gaussian_process as skgp
import matplotlib.pyplot as plt
from gp_edge_tracing import sklearn_gpr
from gp_edge_tracing import gpet_utils
import scipy
import KDEpy 

class GP_Edge_Tracing(object):
    '''
    This module traces an individual edge in an image using Gaussian process regression.
    '''
    
    def __init__(self, 
                 init, 
                 grad_img, 
                 kernel_options=(1,3,3), 
                 noise_y=1, 
                 obs=np.array([]),
                 N_samples=500, 
                 score_thresh=1, 
                 delta_x=10, 
                 keep_ratio=0.1, 
                 pixel_thresh=2,
                 seed=42,
                 return_std=False, 
                 fix_endpoints=False):
        '''
        Internal parameters of the edge tracing algorithm.
        
        INPUTS:
        -------------------
            init (2darray, compulsory input) : Pixel coordinates of the edge of interest. This is 
                inputted in xy-space, i.e. init=np.array([[st_x, st_y],[en_x, en_y]]).

            grad_img (2darray, compulsory input) : Estimated edge map, or image gradient. This is 
                normalised to [0,1]. 

            kernel_options (dict or 3-tuple, compulsory input) : Kernel type and hyperparameters. Only
                the square exponential and Matern class are implemented. Format of dict is, for example,
                {'kernel_type':'RBF', 'sigma_f':50, 'length_scale':100} for the square exponential kernel.
                If inputting 3-tuple, format is (k, s, l) where k=0,1 (RBF, Matern), s,l=0,...5, define
                scales of function variance and lengthscale, chosen according to the image height and edge 
                of interest length, respectively.
            
            noise_y (float, default 1) : Amount of noise to perturb the posterior samples drawn from 
                Gaussian process from the observation set.

            obs (2darray, default np.array([])) : xy-space observations to fit to the Gaussian 
                process. User-specified at initialisation when propagating edge pixels from a different edge 
                to improve accuracy and quicken convergence of the current edge of interest (see section 4.4 
                of paper)). Otherwise, this parameter is used in the recursive Bayesian scheme to converge 
                posterior predictive distribution to edge of interest.

            N_sample (int, default 1000) : Number of posterior curves to sample from the Gaussian 
                process in each iteration. 

            score_thresh (float, default 1) : Initial pixel score to determine which observations to 
                accept and fit to the Gaussian process in each iteration.

            delta_x (int, default 5) : Length of sub-intervals to split horizontal, $x$-axis during 
                search for edge pixels. Number of subintervals=N//delta_x, where $N$ is the width of the 
                image. Algorithm terminates after N//delta_x are located and fitted to Gaussian process.

            keep_ratio (float, default 0.1) : Proportion of posterior curves, coined as optimal, 
                to use to score pixels and select new observations for next iteration. Value must be in (0, 1].
                
            pixel_thresh (int, default 2) : Threshold to reduce score threshold if number of new pixel candidates 
                doesn't exceed the previous number of pixel candidates + pixel_thresh.

            seed (int, default 1) : Integer to fix random sampling of posterior curves from Gaussian Process
                in each iteration.   
            
            return_std (bool, default False) : If flagged, return 95% credible interval as well as edge trace.
            
            fix_endpoints (bool, default False) : If flagged, user-inputted edge endpoints init are fixed 
                (negligible observation noise is chosen for these pixels).
        '''
        
        # Set global internal parameters
        self.init = init.astype(int)
        self.x_st, self.x_en = int(init[0,0]), int(init[-1,0])
        self.grad_img = normalise(grad_img, minmax_val=(0,1), astyp=np.float32)
        self.noise_y = 100 * noise_y
        self.N_samples = int(N_samples) if N_samples > 1 else 1000
        self.obs = obs.reshape(-1,2).astype(np.int8)
        self.seed = seed
        self.keep_ratio = float(keep_ratio) if 0 < keep_ratio <= 1 else 0.1
        self.pixel_thresh = int(pixel_thresh) if pixel_thresh >= 2 else 2
        self.score_thresh = float(score_thresh) if 0 < score_thresh <= 1 else 1
        self.delta_x = int(delta_x) if delta_x > 3 else 2
        self.return_std = return_std
        self.fix_endpoints = fix_endpoints
        self.kde_thresh = 1e-3
        
        # Set up gloabl internal parameters using inputs.
        self.N_inits = self.init.shape[0]
        self.M, self.N = grad_img.shape
        self.x_grid = self.x_st + np.arange(self.x_en - self.x_st + 1).astype(int)
        self.X =  np.repeat(self.x_grid.reshape(-1,1), self.N_samples, axis=-1)
        self.edge_length = self.x_grid.shape[0]
        self.N_subints = int(self.edge_length // self.delta_x)
        self.N_keep = int(keep_ratio*N_samples)
        
        # Compute interpolated gradient surface and KDE of image gradient
        self.grad_interp = scipy.interpolate.RectBivariateSpline(np.arange(self.M), 
                                                                np.arange(self.N), 
                                                                self.grad_img, 
                                                                kx=1, ky=1)
        #self.grad_interp = self.grad_interpolation()
        self.grad_kde = self.kernel_density_estimate(best_curves=None, costs=None)
        
        # Set up check for kernel, function variance and length scale
        if type(kernel_options) == dict:
            self.sigma_f = kernel_options['sigma_f']
            self.sigma_l = kernel_options['length_scale']
            self.kernel_type = kernel_options['kernel']
            self.kernel_nu = kernel_options['nu'] if kernel_options['kernel']=='Matern' else 2.5
        else:
            rbf_matern, sigmaf_opt, sigmal_opt = kernel_options

            # RBF or Matern
            # if rbf_matern = 0 we use RBF kernel, 
            # if rbf_matern = 1, we use Matern kernel with nu = 2.5 (smoother curves, higher degree of differentiability)
            # if rbf_matern >= 2, we use Matern kernel with nu = 1.5 (bumpy curves, lower degree of differentiability)
            self.kernel_type = ["RBF", "Matern"][int(rbf_matern > 0)]
            self.kernel_nu = [2.5, 1.5][int(rbf_matern > 1)]
            
            # Function variance
            sigma_f_const = [10, 8, 6, 4, 2][sigmaf_opt-1] if (sigmaf_opt >= 0) and (sigmaf_opt <= 4) else 2
            self.sigma_f = self.M // sigma_f_const 
            
            # Lengthscale
            sigma_l_const = [1, 4/3, 2, 4, 10][sigmal_opt-1] if (sigmal_opt >= 0) and (sigmal_opt <= 4) else 10
            self.sigma_l = self.edge_length // sigma_l_const    
            
        # Set up default GP parameters during fitting (before converged=True)
        self.gp_params = dict(normalize_y=True, 
                              alpha=1e-10, 
                              copy_X_train=True, 
                              optimizer=None,
                              n_restarts_optimizer = 0)
        
        # Set up initial alpha
        alpha_const = [1.0, 1e-7][int(fix_endpoints)]
        self.alpha_init = alpha_const*np.ones((self.N_inits))
                                   
        # Set up constant kernel
        self.constant_kernel = skgp.kernels.ConstantKernel(self.sigma_f**2, 
                                                           constant_value_bounds="fixed")
        
        # Set up main GP kernel (RBF or Matern)
        if self.kernel_type == 'Matern':
            self.gp_kernel = skgp.kernels.Matern(nu = self.kernel_nu, 
                                                length_scale=self.sigma_l, 
                                                length_scale_bounds="fixed")
        else:
            self.default_kernel = skgp.kernels.RBF(length_scale=self.sigma_l, 
                                                   length_scale_bounds="fixed")
            
        # Define default kernel during fitting procedure (before convergence=True)
        self.default_kernel = self.constant_kernel * self.gp_kernel
        
        
                                    
    def fit_predict_GP(self, 
                       obs, 
                       converged=False, 
                       seed=1):
        ''' 
        Fits a Gaussian Process using init and obs. The coordinates in obs are perturbed using observation 
        noise, self.noise_y. N_samples posterior curves are sampled and outputted (and can be fixed using seed). 
            
        INPUTS:
        -------------------
            obs (2darray) : xy-space observations to fit to the Gaussian process regressor. 
            
            converged (bool, default False) : If each sub-interval has an accepted pixel coordinate,
                then converged=True and we optimise model by minimising the log marginal likelihood, 
                outputting mean function and 95% credible interval to user.
            
            seed (int, default 1) : Seed to fix random posterior curve sampling.
            
        RETURNS:
        -------------------
            y_samples (np.array) : Outputted when converged=False. N_samples posterior curves sampled from
                the Gaussian process posterior. Shape is (N, N_samples, 2).
            
            y_mean, y_std (np.array, np.array) : Outputted when converged=True. Mean and standard deviation of 
                the posterior distribution of the Gaussian process.
        '''
        # Construct inputs and outputs, depending on if endpoints are fixed and if we've converged
        alpha_obs = np.ones((obs.shape[0]))
        alpha = np.concatenate([self.alpha_init, alpha_obs], axis=0)
        new_obs = np.concatenate([self.init, obs], axis=0)                     
        
        # Set up elements of kernel and optimisation parameters for GPR
        self.gp_params["random_state"] = seed
        noise_kernel = sklearn_gpr.WeightedWhiteKernel(edge_length = self.edge_length, 
                                                       noise_weight = alpha, 
                                                       noise_level = self.noise_y, 
                                                       noise_level_bounds = "fixed")
        if not converged:
            iter_kernel = self.default_kernel + noise_kernel
        else:
            self.gp_params['optimizer'] = 'fmin_l_bfgs_b'
            self.gp_params['n_restarts_optimizer'] = 5
            self.gp_kernel.length_scale_bounds = (1e-3*self.sigma_l, 1e3*self.sigma_l)
            self.constant_kernel.constant_value_bounds = (1e-3*self.sigma_f, 1e3*self.sigma_f)
            noise_kernel.noise_level_bounds = (1e-7, self.noise_y)
            iter_kernel = self.constant_kernel * self.gp_kernel + noise_kernel

        # Construct Gaussian process regressor
        self.gp_params['kernel'] = iter_kernel
        gp = sklearn_gpr.GaussianProcessRegressor(**self.gp_params)
    
        # Fit data to the GP to update mean and covariance matrix
        X = new_obs[:,0][:, np.newaxis]
        y = new_obs[:,1]
        gp.fit(X, y)
        
        # Sample posterior curves
        if not converged:
            y_samples = gp.sample_y(self.x_grid[:, np.newaxis], self.N_samples, random_state=seed)
            outputs = y_samples
        else:
            y_mean, y_std = gp.predict(self.x_grid[:, np.newaxis], return_std=True)
            outputs = (y_mean, y_std)

        return outputs
    
    
    
    def grad_interpolation(self, 
                           gmin=1e-12):
        '''
        Linearly interpolates the image gradient to evaluate at real-valued coordinates.
        
        Note that this function takes longer to compute, but quicker to evaluate during
        runtime than RectBivariateSpline. scipy.interpolate.interp2d() is not used as
        evaluation during runtime is very slow, even though fitting interpolant is quick.
        
        # Parameters
        N_samples = 500
        random_trace = (self.M-1)*np.random.random(size=self.N)
        x_grid = np.arange(self.N)
        
        # For LinearNDInterpolator
        %%timeit
        f = grad_interpolation()
        >> 17.4 s ± 1.58 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        %%timeit
        for i in range(N_samples):
            evl = f(x_grid, random_trace)
        >> 16 s ± 636 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        
        # For RectBivariateSpline
        %%timeit
        g = RectBivariateSpline(np.arange(self.M), np.arange(self.N), self.grad_img, kx=1, ky=1)
        >> 60.3 ms ± 2.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        
        %%timeit -n 5 -r 10
        for i in range(N_samples):
            evl = g(random_trace, x_grid, grid=False)
        >> 370 ms ± 27 ms per loop (mean ± std. dev. of 10 runs, 5 loops each)
        
        
        INPUTS:
        -------------------
            gmin (float, default 1e-12) : Values to fill zeros in gradient image.
            
        RETURNS:
        -------------------
            grad_interp (object) : Interpolated gradient surface.
        '''

        # Create meshgrid of integer points denoting indices of image
        x = np.arange(self.N)
        y = np.arange(self.M)

        # Create meshgrid
        X, Y = np.meshgrid(x, y)
        XY = np.stack([X.ravel(), Y.ravel()]).T

        # The surface here is the gradient image. Mark any 0-values to gmin
        self.grad_img[self.grad_img == 0] = gmin
        Z = self.grad_img.ravel()

        # Perform basic linear 2-dimensional interpolation to create gradient surface. 
        # Ensure that if evaluating at point outside input domain, set value to 0
        grad_interp = scipy.interpolate.LinearNDInterpolator(XY, Z, fill_value = 0)

        return grad_interp 
    
    

    def finite_diff(self, 
                    y, 
                    typ=0, 
                    h=1):
        '''
        Function to approximate derivative of function using finite difference methods. A choice of central, forward
        and backward methods are available.

        INPUTS:
        -------------------
            y (array-like) : Array of function values sampled at discrete time points

            typ (str, defalt 'forward') : Type of finite difference method. Assumes forward differencing. 
                If 0, then forward differencing is used (default), if 1 then backward differencing is used,
                if 2 then central differencing is used.
                
           h (int) : Stepsize in the differencing.
           
        RETURNS:
        -------------------
            diff (np.array) : Approximate derivative of edge.
            
        '''
        # Compute differencing depending on type.  Stepsize h is assumed to be 1.
        N = y.shape[0]
        diff = np.empty((N-1), dtype=np.float64)
        lower, upper = [(0, N-1), (1, N), (1, N-1)][typ]
        b, a = [(h, 0), (0, -h), (-h, h)][typ]
        for i in range(lower, upper):
            diff[i-lower] = y[i+b] - y[i+a]

        return diff
    
    
        
    def cost_funct(self,
                   edge):
        '''
        Computes the line integral per unit pixel length along the posterior curve, or candidate edge, given 
        the image gradient.
        
        We numerically integrate the edge by computing the curvilinear coordinates along the image gradient 
        using euclidean distance and Simpson's rule. This is then normalised by the length of the edge to
        penalise sharp movements.

        INPUTS:
        -------------------
            edge (np.array) : edge indices in xy-space, shape=(N, 2), i.e. 
                new_edge = np.array([[x0, y0],...[xN, yN]])
                
        RETURNS:
        -------------------
            cost (np.float64) : Cost of the edge.
        '''
        # Evaluate edge along interpolated gradient image
        edge = edge[edge[:,0].argsort(), :]
        grad_score = self.grad_interp(edge[:,1], edge[:,0], grid=False)
        #grad_score = self.grad_interp(edge[:,0], edge[:,1])

        # Compute cumulative sum of euclidean distance between pixel indexes of edge.
        # This is the equivalent of computing the curvilinear coordinates of the edge
        pixel_diff = np.cumsum(np.sqrt(np.sum(np.diff(edge, axis=0)**2, axis=1)))

        # Compute integrand of arc length of edge
        pixel_deriv = self.finite_diff(edge[:,1], typ=0, h=1)
        integrand = np.sqrt(1 + pixel_deriv**2)

        # Compute cost = 1 / (line integral / arc length) = arc length / line integral - smaller costs are better 
        line_integral = scipy.integrate.simps(grad_score[:-1], pixel_diff)
        arc_length = scipy.integrate.simps(integrand, edge[:-1,0])
                                    
        # Cost is the line integral per unit pixel length
        cost = arc_length / line_integral 
        
        return cost
    
    
    
    def get_best_curves(self, 
                        y_samples):
        '''
        This function works out the optimal posterior curves sampled from the Gaussian process posterior 
        predictive distribution of the current iteration.

        INPUTS:
        -------------------
            y_samples (np.array) : N_samples posterior curves sampled from the gaussian process posterior of 
                the current iteration. Shape is (N, N_samples, 2).
            
        RETURNS:
        -------------------
            best_curves (np.array) : Top N_keep posterior curves with the most optimal costs.

            best_costs (np.array) : Array of costs for the N_keep posterior curves.
            
            optimal_curve, optimal_cost : Top posterior curve of the iteration and its cost.
        '''
        # Stack the samples and their input locations to be fetched for computing their cost.
        curves = np.stack((self.X, y_samples), axis=-1)

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
    
        
        
    def kernel_density_estimate(self,
                                best_curves, 
                                costs,
                                bw=1):
        '''
        This function estimates the kernel density function of the Gaussian process posterior predictive
        distributions' belief in where the edge of interest lies, through using the finite sample distribution 
        of the most optimal posterior curves. 
        
        If best_curves and costs are None, we compute the KDE of the image gradient.

        INPUTS:
        -------------------
            best_curves (np.array, default None) : Best curves from iteration stored in an array of 
                shape (N, N_keep, 2). Default is None as this functon is used to estimate image gradient 
                weighted smoothing kernel for pixel scoring.

            costs (np.array) : Costs of each of the optimal posterior curves. Default is None as
                this functon is used to estimate image gradient weighted smoothing kernel for pixel scoring.

            bw (float, default 1) : Bandwidth of Gaussian kernel. Assumed to be 1 so we have the lengthscale equal to
                approximately a pixel. This is so each point on a posterior curve to contribute to at most 4 nearby
                pixel coordinates in the discretised space.
            
        RETURNS:
        -------------------
            disc_kde (np.array) : Kernel density estimate of either the top N_keep posterior curves or the image
                gradient. Shape is (self.M, self.N).
        '''
        # Compute KDE of curve intersection
        if costs is not None:
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
                                    
        # Compute KDE of image gradient
        else:
            # Extract non-zero gradient intensity pixel coordinates.
            sample_pts = np.argwhere(self.grad_img > self.kde_thresh)
                                    
            # Weight each coordinate according to their gradient intensity
            weights_arr = self.grad_img[sample_pts[:,0], sample_pts[:,1]].reshape(-1)
            sample_pts = sample_pts[:, [1,0]].reshape(-1,2)

        # Estimate kernel density function using a Gaussian kernel with identity bandwidth matrix
        # This is equivalent to centering a Gaussian kernel so we have a pixel lengthscale and no dominant 
        # orientation
        gauss_kernel = KDEpy.FFTKDE(kernel='gaussian', bw=bw).fit(sample_pts, weights=weights_arr)
        x_range = np.arange(-1, self.N+1)
        y_range = np.arange(-1, self.M+1)
        mesh_xy = np.meshgrid(x_range, y_range)
        xy_range = np.stack([mesh_xy[0].T.ravel(), mesh_xy[1].T.ravel()]).T

        # Evaluate KDE along the pixel coordinates of the image
        disc_kde = gauss_kernel.evaluate(xy_range).reshape((self.N+2, self.M+2)).T
        disc_kde = disc_kde[1:-1, 1:-1]

        # Normalise PDF between 0 and 1 rather than leave it as a well-defined PDF (where 
        # integrating across the domain equals 1). This is so we can score pixels ensuring their 
        # score is in [0,1].
        disc_kde = gpet_utils.normalise(disc_kde, (0,1), np.float64)

        return disc_kde   
        
        
        
    def bin_pts(self, 
                best_pts_scores):
        '''
        This bins the best pixels into sub-intervals splitting, performing non-max suppression
        to select the highest scoring pixel in each subinterval. These pixel coordinates
        will be fitted to the Gaussian process in xy-space for the next iteraton.

        INPUTS:
        -------------------
            best_pts_scores (np.array) : Array of pixel coordinates whose score surpasses the score threshold.
                First and second axes represents x- and y-coordinates of points and third axis represents the
                corresponding score.
                
        RETURNS:
        -------------------
            fobs (np.array) : Highest scoring pixels, one pixel per sub-interval.
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

        return fobs
        
        
        
    def comp_pixel_score(self, 
                         pixel_idx, 
                         kde_arr, 
                         pre_fobs):
        '''
        This function computes the scores for all the pixels which have had a large enough density
        on the kernel density estimate from the optimal posterior curves. It also scores pixels 
        found in previous iterations and combined old and new. Note, some if not most of these older
        pixels may be discarded if newer ones have higher scores in the corresponding sub-intervals.
        
        Pixel candidates are those whose scores exceed the score threshold. While the number of pixel
        candidates doesn't exceed the number of previously dicovered candidates, reduce score
        threshold and repeat.

        INPUTS:
        -------------------
            pixel_idx (np.array) : Array of pixel indexes which had a non-zero density on the KDE of the 
                optimal posterior curves. In practice, this is a density greater than self.kde_thresh.

            kde_arr (np.array) : Kernel density function of optimal posterior curves.

            pre_fobs (array) : Array of accepted fixed points from previous iteration. 
                Shape: (N_fixed_pts, 2). These are rescored with the new pixel candidates.
                
        RETURNS:
        -------------------
            fobs (np.array) : Highest scoring pixels, one pixel per sub-interval.
        '''
        # Number of previously fitted observations
        N_pixels_pre = pre_fobs.shape[0]
                                    
        # Compute their gradient value and normalized intersection score of the best curves
        new_grad_vals = self.grad_kde[pixel_idx[:,0], pixel_idx[:,1]]
        new_int_vals = kde_arr[pixel_idx[:,0], pixel_idx[:,1]]

        # Compute new scores for previous observations
        #old_int_vals = kde_arr[pre_fobs[:,0], pre_fobs[:,1]]
                                    
        # Remove those which have no intersections with best curves of new iteration
        #old_int_idx = old_int_vals > self.kde_thresh
        #old_fobs = pre_fobs[old_int_idx]
        #old_int_vals = old_int_vals[old_int_idx]
        #old_grad_vals = self.grad_kde[old_fobs[:,0], old_fobs[:,1]] 
                                    
        # Concatenate old and new int and grad vals
        #pixel_candidates = np.concatenate([old_fobs, pixel_idx], axis=0)
        #intersection_vals = np.concatenate([old_int_vals, new_int_vals], axis=0)   
        #grad_vals = np.concatenate([old_grad_vals, new_grad_vals], axis=0)    
        pixel_candidates = pixel_idx
        intersection_vals = new_int_vals
        grad_vals = new_grad_vals
        
        # Compute score for each pixel based on KDE's of image gradient and posterior curves. 
        pixel_scores = 1/3 * (intersection_vals * grad_vals + intersection_vals + grad_vals)
        
        # Initialise number of new pixels. If the new set of pixel coordinates is smaller 
        # than the previous set (or is too small for first iteration) reduce score and try
        # find more pixel coordinates with lower score.
        N_pixels = N_pixels_pre
        i = 0
        while N_pixels - N_pixels_pre < self.pixel_thresh:
            
            # Threshold pixel scores according to threshold
            reduce_score_flag = int(i == 0)
            self.score_thresh *= [0.95, 1.0][reduce_score_flag]
            best_mask = pixel_scores >= self.score_thresh
        
            # Concatenate best pixels and scores by thresholding scores. Ensure to switch columns of 
            # pixels to turn them from pixel-space (yx-space) to xy-space.  
            best_pixels = pixel_candidates[best_mask].reshape(-1,2)
            best_scores = pixel_scores[best_mask].reshape(-1,1)
            best_pts_scores = np.concatenate((best_pixels[:,[1,0]], best_scores), axis=1)

            # Bin best points into sub-intervals of length delta_x and choose highest scoring point 
            # per sub-interval. This performs the binning and non-max suppression procedure.
            fobs = self.bin_pts(best_pts_scores)
            N_pixels = fobs.shape[0]
            i += 1
                                   
        return fobs   

    
    
    def get_best_pixels(self, 
                        best_curves, 
                        costs, 
                        pre_fobs):
        '''
        This function determines the next set of fixed points to fit to the Gaussian Process. 
        It will estimate the density function of the optimal posterior curves, score the pixel 
        coordinates which are large enough to be considered potential candidates, bin and perform 
        non-max suppression to obtain a pixel coordinate for each subinterval.

        INPUTS:
        -------------------
            best_curves (array) : Optimal posterior curves from iteration stored in an array of 
                shape: (N, N_samples*keep_ratio, 2)

            costs (array) : Costs of each of the optimal posterior curves.
            
            pre_fobs (array) : Array of accepted fixed points from previous iteration. 
                Shape: (N_fixed_pts, 2). These are rescored with the new pixel candidates.
            
        RETURNS:
        -------------------
            fobs (np.array) : Highest scoring pixels, one pixel per sub-interval.
        '''
        # Compute the density function representing the frequency distribution of optimal posterior 
        # curves exploring the image
        kde_arr = self.kernel_density_estimate(best_curves, costs)

        # Only store those pixel coordinates as candidates if their density is "reasonably" non-zero 
        pixel_idx = np.argwhere(kde_arr > self.kde_thresh)

        # If fixing endpoints, then assume endpoints are true estimates of the edge and don't accept
        # any new pixels for these columns.
        if self.fix_endpoints:
            pixel_idx = pixel_idx[(pixel_idx[:,1] > self.x_st) & (pixel_idx[:,1] < self.x_en)]

        # Compute pixel score and threshold pixels using score_thresh
        fobs = self.comp_pixel_score(pixel_idx, kde_arr, pre_fobs)

        return fobs
  
    
                                    
    def plot_iter(self, 
                  y_samples, 
                  N_plt_samples, 
                  obs):
        '''
        If flagged, plot a subsample of the posterior curves, estimated mean and std deviation curves,
        initial pixels and observations, if any.
        
        INPUTS:
        -------------------
            y_samples (array) : Array of posterior curves sampled from gaussian process 
                of previous iteration. Shape: (N, N_samples)
        
            plt_post (bool, default False) : Flag to plot N_plt_samples curves sampled from posterior 
                predictive distribution. 

            N_plt_samples (int, default 20) : Number of curves to sample from Gaussian process to sample 
                if plt_post=True. 
        '''
        # Extract N samples, estimate mean and std curves
        y_plt_samples = y_samples[:,:N_plt_samples]
        y_mean_est = np.mean(y_samples, axis=1)
        y_std_est = np.std(y_samples, axis=1)
        fontsize = 16
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Plot mean function, 95% confidence interval and N_plt_samples curves
        ax.plot(self.x_grid, 
                y_mean_est, 
                c='k', lw=3, zorder=3, label='Posterior Predictive Mean')
        ax.fill_between(self.x_grid, 
                        y_mean_est - 1.96*y_std_est,
                        y_mean_est + 1.96*y_std_est, 
                        alpha=0.2, color='k', zorder=1, label='95% Credible Region')
        ax.plot(self.x_grid, 
                y_plt_samples, 
                lw=1, zorder=2)

        # Overlay initial endpoints and observations, if any
        ax.scatter(self.init[:, 0], 
                   self.init[:,1], 
                   c='m', s=5*fontsize, zorder=5, edgecolors=(0, 0, 0), label='Edge Endpoints')
        if obs.size > 0:
            ax.scatter(obs[:,0], 
                       obs[:,1], 
                       c='r', s=3*fontsize, zorder=4, edgecolors=(0, 0, 0), label='Observations')
            
        # Maptlotlib parameters
        ax.set_xlim([0, self.N-1])
        ax.set_ylim([self.M-1, 0])
        ax.set_xlabel('Pixel Column, $x$', fontsize=fontsize)
        ax.set_ylabel('Pixel Row, $y$', fontsize=fontsize)            
        handles, labels = ax.get_legend_handles_labels()
        ax_legend = ax.legend(handles, labels, fontsize=10, ncol=2, loc='lower right', edgecolor=(0, 0, 0, 1.))
        ax_legend.get_frame().set_alpha(None)
        ax_legend.get_frame().set_facecolor((1,1,1,1))
        fig.tight_layout()
        plt.show()

    
    
    def plot_diagnostics(self, 
                         iter_optimal_curves, 
                         iter_optimal_costs, 
                         credint=None):
        '''
        Plot the optimal posterior curves on top of the image gradient. Plot a graph of the 
        costs against iteration.

        INPUTS:
        -------------------
            iter_optimal_curves (list) : List of optimal curves for each iteration.

            iter_optimal_costs (list) : List of costs of optimal curves for each iteration.

            credint (2darray, default None) : Upper and lower boundaries of 95% credible interval.
                If None, then credible interval not plotted.
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
    
    
    
    def __call__(self, 
                 print_final_diagnostics=False, 
                 show_init_post=False, 
                 show_post_iter=False, 
                 verbose=False):
        '''
        Call function that runs the Gaussian process edge tracing algorithm.

        INPUTS:
        ---------------
            print_final_diagnostics (bool, default False) : If flaged, print diagnostics 
                at the end; plot of edge prediction with/without 95% credible interval on  
                image gradient and plot of iteration vs. cost.

            show_init_post (bool, default False) : If flagged, show initial posterior predictive 
                distribution (after fitting init (and obs) and ask use to continue or not.
            
            show_post_iter (bool, default False) : If flagged, show the posterior predictive 
                distribution during each iteration of the fitting procedure.

            verbose (bool, default False) : If flagged, output text updating user on time/iter, 
                number of observations/iter and any adaptive score threshold reduction.
                
        RETURNS:
        ---------------
            edge_trace (np.array, dtype=np.int) : Predicted edge in yx-pixel space.
            
            cred_interval (np.array, dtype=np.float64) : 95% credible interval of predicted edge trace.
        '''
        # Plot random samples drawn from Gaussian process with chosen kernel and ask to continue with algorithm.   
        if show_init_post:
            y_samples = self.fit_predict_GP(self.obs, converged=False, seed=0)
            self.plot_iter(y_samples, 20, self.obs)
            print('Are you happy with your choice of kernel? y/n')
            cont = input()
            if cont.lower()[0] != 'y':
                return

        # Measure time elapsed
        alg_st = t.time()
        
        # Initialise the number of sub-intervals dividing the x-axis, the number of observations 
        # added to the Gaussian process and the two lists storing the optimal curve and its cost 
        # in each iteration
        pre_fobs = self.obs
        n_fobs = pre_fobs.shape[0]
        iter_optimal_curves = []
        iter_optimal_costs = []

        # Convergence of the algorithm is when there is an observation fitted for each sub-interval. 
        # In this implementation, due to delta_x, I've set the While loop to stop when there is an 
        # observation fitted for all sub-intervals bar one.
        N_iter = 1
        while n_fobs < self.N_subints-(self.pixel_thresh-1):
            # Start timer for iteration
            st = t.time()

            # If verbose, let user know that GP is being fit and new observations are being searched for
            if verbose:
                print('Fitting Gaussian process and computing next set of observations...')

            # Intiialise Gaussian process by fitting initial points, outputting samples drawn from initial posterior 
            # distribution 
            y_samples = self.fit_predict_GP(pre_fobs, converged=False, seed=self.seed+N_iter)
            
            # Plot posterior curves if show_post_iter == True 
            if show_post_iter:
                self.plot_iter(y_samples, 20, pre_fobs)

            # Compute optimal posterior curves via cost function, returning N_keep of the best curves. 
            best_curves, best_costs, (optimal_curve, optimal_cost) = self.get_best_curves(y_samples)
            iter_optimal_curves.append(optimal_curve)
            iter_optimal_costs.append(optimal_cost)
                                    
            # Determine the set of pixel coordinates which exceed the score threshold for the current iteration. 
            # Scores are computed using the image gradient and optimal posterior curves. We make sure to have
            # at least 1 *more* pixel fitted in consecutive iterations by reducing score threshold by 5% if necessary
            pre_fobs = self.get_best_pixels(best_curves, best_costs, pre_fobs)
            
            # Recompute number of observations
            n_fobs = pre_fobs.shape[0]

            # Incremement number of iterations and output time taken to perform iteration
            en = t.time()
            N_iter += 1

            # If verbose, print out number of observations 
            if verbose:
                print(f'Number of observations: {n_fobs}')
                print(f'Iteration {N_iter} - Time Elapsed: {round(en-st, 4)}\n\n')

        # Once condition above is met, we optimise the observation noise and kernel hyperparameters by maximising 
        # the marginal likelihood of the training set of pixel coordinates. 
        output = self.fit_predict_GP(pre_fobs, converged=True, seed=self.seed+N_iter)
        y_mean_optim, y_std = output
        cred_interval = (y_mean_optim - 1.96*y_std, y_mean_optim + 1.96*y_std)
        # Note, this doesn't take into account the optimisation of these pixel coordinates being edge coordinates 
        # of the edge of interest however. Future work would include this. 
            
        # Compute predicted edge by converting mean function to pixel coordinates
        optim_mean_curve = np.concatenate([self.x_grid[:,np.newaxis], y_mean_optim[:, np.newaxis]], axis=1)
        edge_trace = np.rint(optim_mean_curve[:, [1,0]]).astype(int)

        # Print diagnostics 
        if print_final_diagnostics:
            iter_optimal_curves.append(edge_trace[:, [1,0]])
            iter_optimal_costs.append(self.cost_funct(optim_mean_curve))
            self.plot_diagnostics(iter_optimal_curves, iter_optimal_costs, cred_interval)

        # Compute and print time elapsed
        alg_en = t.time()
        if verbose:
            print(f'Time elapsed before algorithm converged: {round(alg_en-alg_st, 3)}') 

        # if returning 95% credible interval
        if self.return_std:
            return edge_trace, cred_interval
        else:
            return edge_trace