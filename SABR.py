# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:34:08 2020

@author: nhian
"""
# =============================================================================
# Normal SABR
# =============================================================================
import numpy as np
from scipy import optimize
import warnings, sys
import matplotlib.pyplot as plt



def SABR_normal_vol(f, k, tau, alpha, beta, rho, volvol):
    """
    Given SABR parameters, return Normal Impl Vol
    """

    if abs(f-k) <= 1e-5:
        impl_vol = alpha * f**beta *(1 + \
                                (beta * (beta -2))/24 * alpha**2 /((f)**(2 - 2 *beta)) * tau +
                                0.25 * (alpha * beta * rho * volvol)/( (f)**((1 - beta)) ) * tau +
                                (2 - 3 * rho**2)/24 * volvol**2 * tau)
    
    else:
        f_mid = (f+k)/2
        z = volvol/alpha * (f**(1-beta) - k**(1-beta))/(1-beta)
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2 ) - rho + z )/(1 - rho))
        expansion_term_1 = 1 + (beta * (beta -2))/24 * alpha**2 /((f_mid)**(2 - 2 *beta)) * tau
        expansion_term_2 = 0.25 * (alpha * beta * rho * volvol)/( (f_mid)**((1 - beta)) ) * tau
        expansion_term_3 = (2 - 3 * rho**2)/24 * volvol**2 * tau
                
        impl_vol = volvol * (f-k)/x_z * (expansion_term_1 + expansion_term_2 + expansion_term_3)

    return(impl_vol)



def atm_sigma_to_alpha(f, tau, atm_vol_n, beta, rho, volvol):
    '''
    Returns alpha given the the at-the-money volatility
    '''
    p0 = 1/24 * (beta *(beta - 2)**2) / (f**(2- 2*beta)) * tau
    # p0 = 1/24 * ((1 - beta)**2) / (f**(2- 2*beta)) * tau
    p1 = 0.25 * (beta * rho * volvol)/(f**(1-beta) ) * tau
    p2 = (1+ (2 - 3 * rho**2)/24 * volvol**2 * tau)
    p3 = -atm_vol_n * f**(-beta) #f**(-beta)
    coeffs=[p0, p1, p2, p3]
    
    r = np.roots(coeffs)    #find the roots of the cubic equation
    
    # real_r = r* (r>0) + sys.float_info.max * (r<0 | abs(r.imag > 1e-6))
    # np.min(real_r)
    return r[r.real >= 0].real.min()



def SABR_first_guess(f, tau, beta, rho, nu, market_n_vols, market_strikes):
    '''
    Explicit SABR Calibration
    Fabien Le Floc'h and Gary Kennedy (2014)
    market_n_vols = 3x1 array of market vols vol-, vol_atm, vol+
    market_strikes = 3x1 array of corresponding market strikes
    The expansion uses log moneyness -> no negative moneyness
    
    f = current forward
    tau = option expiry
    beta = beta guess
    rho = SABR rho guess
    nu = SABR nu guess
    
    '''
    vol_min, vol0, vol_plus = market_n_vols
    #z = log moneyness    
    z_min, z0, z_plus = np.log(market_strikes/f)
    
    w_min = 1/((z_min - z0)*(z_min - z_plus))
    w0 = 1/((z0 - z_min)*(z0 - z_plus))
    w_plus = 1/((z_plus - z_min)*(z_plus - z0))
    
    #3 point finite difference
    sigma_0 = z0 * z_plus * w_min * vol_min + \
                z_min * z_plus * w0 * vol0 + \
                z_min * z0 * w_plus * vol_plus
    
    #first derivative
    #sigma_0'
    skew = -(z0 + z_plus) * w_min * vol_min - \
            (z_min + z_plus) * w0 * vol0 - \
            (z_min + z0) * w_plus * vol_plus
    
    #second derivative
    #sigma_0''
    curveness = 2 * w_min * vol_min + 2 * w0 * vol0 + 2 * w_plus * vol_plus
    
    alpha_0 = sigma_0 * (f)**(-beta)
    nu_sq = 1/(f**2) * (3 * sigma_0 * curveness - \
                          0.5 * (beta**2 + beta) * sigma_0**2 - \
                          3 * sigma_0 *(skew - 0.5 * beta * sigma_0) +\
                          1.5 * (2 * skew - beta * sigma_0)**2)
    
    if nu_sq < 0:
        rho = np.sign(2 * skew + (1-beta) * sigma_0)
        nu = 1/rho * (2 * skew + (1-beta) * sigma_0)
    else:
        nu = np.sqrt(nu_sq)
        rho = 1/(nu * f) * (2 * skew - beta * sigma_0)
    
    
    #refine alpha guess
    refined_alpha = atm_sigma_to_alpha(f, tau, sigma_0, beta, rho, nu)
    
    return(refined_alpha, rho, nu)
    

def SABR_calibration(f, tau, atm_vol_n, beta, market_strikes, market_vols, guess = None):
    ''' Returns the parameters alpha, nu and rho given a parameter beta, 
    forward price, a list of market volatilities and corrsponding strike 
    spread. Instead of doing a regression in all three parameters, this method 
    calculates alpha when needed from nu and rho. Hence a regression is done 
    in only two variables.
    '''
    
    def func_to_optimize(params):
        rho, volvol = params
        alpha = atm_sigma_to_alpha(f, tau, atm_vol_n, beta, rho, volvol)
        diff = np.zeros(len(market_strikes))
        for i in range(len(market_strikes)):
            diff[i] = market_vols[i] - SABR_normal_vol(f, market_strikes[i],
                                                       tau, alpha, beta, rho, volvol)
        
        return  diff
    if guess == None:
        guess = [0, 0.5]
    sol = optimize.least_squares(func_to_optimize, x0 = guess,bounds = ((-1, 0), (1, np.inf)) )
    
    # sol = optimize.minimize(func_to_optimize, x0 = [0, 0.5], method = "Powell")
    if sol.status == 0:
        warnings.warn("Solution did not converge")
    
    rho, volvol = sol.x
    alpha = atm_sigma_to_alpha(f, tau, atm_vol_n, beta, rho, volvol)
    
    return np.array([alpha, rho, volvol])



# =============================================================================
# Test Case
# =============================================================================
# beta = 0
# f = -0.00174
# ATMStrike = -0.174/100;
# MarketStrikes  = ATMStrike + np.arange(-0.5, 1.75, 0.25)/100
# MarketVolatilities = np.array([20.58, 17.64, 16.93, 18.01, 20.46, 22.90, 26.11, 28.89, 31.91])/10000
# ATMVolatility = MarketVolatilities[2]


# SABR_first_guess(f, 1, 0, -0.00286745,  0.43171191, MarketVolatilities[:3], MarketStrikes[:3])

# sabr_params = SABR_calibration(f, 1, ATMVolatility, 0, MarketStrikes, MarketVolatilities)
# sabr_params = SABR_calibration(f, 1, ATMVolatility, 0, MarketStrikes, MarketVolatilities,
#                                guess = [.5, .89])
# sabr_res = np.zeros(len(MarketStrikes))
# for k in range(len(MarketStrikes)):
#     sabr_res[k] = SABR_normal_vol(f, MarketStrikes[k], 1, sabr_params[0], 0, sabr_params[1], sabr_params[2])



# # =============================================================================
# # test
# # =============================================================================

# beta = 0
# f = 1/100
# ATMStrike = 1/100
# MarketStrikes  = np.array([-0.25, 0, 0.25,	0.5	 ,0.75,	1,	1.25,	1.5])/100
# MarketVolatilities = np.array([75.68, 73.71, 71.75,	69.83,	68.02,	66.38,	65,	63.98])/10000
# ATMVolatility = 66.38/10000

# sabr_params = SABR_calibration(f, 9.5, ATMVolatility, 0, MarketStrikes, MarketVolatilities)

# sabr_res = np.zeros(len(MarketStrikes))
# for k in range(len(MarketStrikes)):
#     sabr_res[k] = SABR_normal_vol(f, MarketStrikes[k], 9.5, sabr_params[0], 0, sabr_params[1], sabr_params[2])

# # Normal_Impvol_Formula(f,b = 0 ,MarketStrikes,alpha,sigma0,0,rho,09.5)

# plt.plot(sabr_res)
# plt.plot(MarketVolatilities)

