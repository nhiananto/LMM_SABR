# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:42:30 2020

@author: nhian
"""
# =============================================================================
# Swaption SABR relation to LMM-SABR Forward
# =============================================================================
import numpy as np
from scipy.integrate import quad, dblquad

def instantaneous_vol(t, expiry, params):
    a, b, c, d = params
    tau = expiry - t
    g = (a + b * tau) * np.exp(-c * tau) + d
    
    return g


expiry_grid = np.arange(1, 20)/2

#swap approximation
#expiry = m
#tenor = n
#expiry_grid
#correction = initial correction for the volatility term (s) (weight freezing only use initial values s0[fwd_k])
def initial_weights(m):
    '''
    beta_f = beta for the forward rate
    beta_swap = beta for swap m x n
    swap = m x n
    #m expiry, n tenor
    '''
    w0 = expiry_grid[m+1] * discount[m+1]/ \
            np.sum(expiry_grid[(m+1) : (n+1)] * discount[ (m+1) : (n+1)]) #from m+1 to n
    w = w0 * initial_forward[m]**beta_f / swap_rate(m, n)**beta_swap
    return w


vol_approx = 0
tm = expiry_grid[m] #expiry of swaption
for i in range(m, n):
    for j in range(m, n):
        ti = expiry_grid[i]
        tj = expiry_grid[j]
        vol_approx  += corr_fwd_fwd[i, j] * weights(i) * weights(j) * correction[i] * correction[j] * \
                        quad(lambda t: deterministic_vol(t, ti, params) * deterministic_vol(t, tj, params), 0, tm)[0]
vol_approx = np.sqrt(1/tm * vol_approx)



volvol_approx = 0
for i in range(m, n):
    for j in range(m, n):
        ti = expiry_grid[i]
        tj = expiry_grid[j]
        integral_term = quad(lambda t: deterministic_vol(t, ti, params) * deterministic_vol(t, tj, params) * \
                 quad(lambda tau: deterministic_volvol(tau, ti, params_h) * deterministic_volvol(tau, tj, params_h), \
                      0, t)[0], #first integral bound #use * t/np.sqrt(ti * tj) or not?
                     0 , tm)[0] #second integral bound
        
        volvol_approx  += corr_fwd_fwd[i, j] * corr_vol_vol[i, j] * weights(i) * weights(j) * correction[i] * correction[j] * \
                            integral_term                        
volvol_approx = 1/(vol_approx * tm) * 2 * volvol_approx




swap_rate_vol_approx = 0
for i in range(m, n):
    for j in range(m, n):
        ti = expiry_grid[i]
        tj = expiry_grid[j]
        integral_term = quad(lambda t: deterministic_vol(t, ti, params) * deterministic_vol(t, tj, params) * \
                 quad(lambda tau: deterministic_volvol(tau, ti, params_h) * deterministic_volvol(tau, tj, params_h), \
                       0, t)[0], #first integral bound
                     0 , tm)[0] #second integral bound
        omega_i_j = (2 * corr_fwd_fwd[i, j] * corr_fwd_vol[i, j] * weights(i) * \
                weights(j) * correction[i] * correction[j])/(volvol_approx * vol_approx * tm)**2
        swap_rate_vol_approx += omega_i_j * corr_fwd_vol[i, j]
