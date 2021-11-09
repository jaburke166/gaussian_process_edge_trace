# -*- coding: utf-8 -*-
"""
Created on Wed 28 September 18:36:12 2021

@author: Jamie Burke
@email: s1522100@ed.ac.uk

This module traces an edge in an image using Gaussian process regression.
"""
from .gpet import GP_Edge_Tracing
from .sklearn_gpr import GaussianProcessRegressor
from . import gpet_utils

__all__ = ['GP_Edge_Tracing', 'GaussianProcessRegressor',
            'gpet_utils']