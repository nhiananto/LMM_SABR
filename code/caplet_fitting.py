# =============================================================================
# Cap stripping
# =============================================================================
import numpy as np
from scipy import interpolate, stats, integrate, optimize
import pandas as pd
import pysabr
import warnings
import math
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from matplotlib import cm

module_path = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/code/"
os.chdir(module_path)

from SABR import *


#read in cap data
path = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/data/string model/newdata/"

cap_df = pd.read_excel(path+"cap_atm.xlsx", sheet_name = 2, header = 0)

cap_mat = np.array(cap_df.iloc[:,:8])

maturities = np.array(cap_df["Maturity"])
fixing_time = maturities - 0.5


# =============================================================================
# read in today's spot rates data(12/3/2020)
# =============================================================================
raw_spot_df = pd.read_csv(path+"spot.csv")
raw_spot_df["Tenor"] = np.array([1/12, 2/12, 3/12, 0.5, 0.75, 1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 20, 30, 50])

#interpolate spot rates data
tenor = np.arange(1, 21)*0.5
cs = interpolate.CubicSpline(raw_spot_df["Tenor"], raw_spot_df["Spot"])
spots = cs(tenor)/100

#discount bonds
zeros = (1 + spots/2)**(- tenor*2)

# forwards = 2 * (zeros[:-1]/zeros[1:] - 1)
forwards = np.array(cap_df["Reset Rate"]/100)




#current forward curve
fig = plt.figure()
plt.plot(fixing_time, forwards)
plt.title('Current 6 mo. Forward Rates')
plt.xlabel('Fixing Time')
plt.ylabel('Forward Rate')


fig = plt.figure()
plt.plot(tenor, spots)
plt.title('Spot Rates')
plt.xlabel('Tenor')
plt.ylabel('Spot Rates')





# =============================================================================
# fill in strikes data
# =============================================================================
strikes_list = [float(x)/100 for i,x in enumerate(cap_df.columns[:8])]
strike_mat = np.repeat(np.array(strikes_list).reshape(-1, 1), cap_mat.shape[0], axis = 1)
strike_mat = strike_mat.T


#compute forward par-rates starting at year 0.5
# (D(N) - D(N+M))/Annuity(N+1:M)
fpar = np.zeros(19)
for i in range(1, 20):
    fpar[i-1] = 2 * (zeros[0] - zeros[i])/np.sum(zeros[1:(i+1)])
    
    
# =============================================================================
# use strike interpolator, fit spline to get ATM vols for each of the caplets
# =============================================================================

atm_vols = np.zeros(cap_mat.shape[0])
for i in range(cap_mat.shape[0]):
    cs = interpolate.CubicSpline(strike_mat[i, :], cap_mat[i, :])
    atm_vols[i] = cs(forwards[i])



#append atm vols and atm strikes
cap_mat = np.hstack((cap_mat, atm_vols.reshape(-1,1)))
strike_mat = np.hstack((strike_mat, forwards.reshape(-1,1)))

# =============================================================================
# pricer and vol functions for caplets
# =============================================================================
#black formula
def black_caplet(f, k, discount, t, fixing, vol):
    '''
    f = forward at fixing date t
    k = strike
    t = start date, default = 0
    fixing = fixing date of forward rate for the caplet
    discount = P(0, T_payment)
    vol = normal implied vol * np.sqrt(fixing - t)
    '''
    delta = 0.5 #semi-annual
    #discount to payment date not fixing date
    expiry = fixing - t
    
    d1 = np.log(f/k)/(vol * np.sqrt(expiry)) +  0.5 * vol*np.sqrt(expiry)
    d2 = np.log(f/k)/(vol * np.sqrt(expiry)) -  0.5 * vol*np.sqrt(expiry)
    price = discount * delta * (f * stats.norm.cdf(d1) - k * stats.norm.cdf(d2))
    return price

#bachelier formula
def bachelier_caplet(f, k, discount, t, fixing, vol):
    '''
    f = forward at fixing date t
    k = strike
    t = start date, default = 0
    fixing = fixing date of forward rate for the caplet
    discount = P(0, T_payment)
    vol = normal implied vol * np.sqrt(fixing - t)
    '''
    delta = 0.5
    
    #discount to payment date not fixing date
    expiry = fixing - t
    
    d = (f - k)/(vol * np.sqrt(expiry))
    price = discount * delta * ((f - k) * stats.norm.cdf(d) + (vol * np.sqrt(expiry)) *
                                          stats.norm.pdf(d))
    # price = discount * delta * (vol * np.sqrt(expiry)) * (stats.norm.pdf(d) +
    #                                       d * stats.norm.cdf(d))
    return price


def black_normal_caplet(f, k, discount, expiry, vol):
    d1 = (f - k)/(vol * np.sqrt(expiry))
    price = discount * ((f - k) * stats.norm.cdf(d1) + \
        vol * np.sqrt(expiry)/np.sqrt(2 * math.pi) * np.exp(- 0.5 *d1**2))
    return(price)
    

#implied vol root finding
def impl_vol(f, k, discount, t, fixing, price, pricer_fn):
    '''
    f = forward at fixing date t
    k = strike
    t = start date, default = 0
    fixing = fixing date of forward rate for the caplet
    discount = P(0, T_fixing)
    price = price of the caplet at t
    price_fn = bachelier/black-76
    '''    
    root = optimize.root_scalar(lambda x: price - 
                  pricer_fn(f = f, k = k, discount = discount, t = t, fixing = fixing, vol = x), bracket = [-100, 100])
    
    # if root.status == 0:
    #     warnings.warn("Solution Did Not Converge")
    
    return root.root



# =============================================================================
# Plots        
# =============================================================================
#plot caplet vol surface
x_axs = np.tile(fixing_time.reshape(-1,1), strike_mat.shape[1])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x_axs, strike_mat*100, cap_mat, cmap = cm.viridis)
ax.view_init(20, 210)
ax.set_xlabel('Fixing Time')
ax.set_ylabel('Strike %')
ax.set_zlabel('Implied Volatility (bps)')
ax.set_title('Caplet Implied Volatility Surface')

##################################################################################


bachelier_sabr_params = np.zeros((cap_mat.shape[0], 5))
for i in range(bachelier_sabr_params.shape[0]):
    #last column is atm
    sabr =  SABR_calibration(forwards[i], tau = fixing_time[i], atm_vol_n = cap_mat[i, -1]/10000,
                              beta = 0, market_strikes = strike_mat[i, :],
                              market_vols = cap_mat[i, :]/10000)

    bachelier_sabr_params[i, 0] = fixing_time[i]
    bachelier_sabr_params[i, 1] = forwards[i]
    bachelier_sabr_params[i, 2:5] = sabr

'''    
#bachelier_sabr_params
column 2: alpha 
column 3: rho
column 4: nu
'''

#recreate vol surface with new calibrated sabr parameters
#volsurface difference
calibrated_impl_vols = np.zeros_like(cap_mat)
for i in range(calibrated_impl_vols.shape[0]):
    for j in range(calibrated_impl_vols.shape[1]):
        calibrated_impl_vols[i, j] = SABR_normal_vol(forwards[i], k = strike_mat[i, j], tau = fixing_time[i],
                                                     alpha = bachelier_sabr_params[i, 2], beta = 0,
                                                     rho = bachelier_sabr_params[i, 3], volvol = bachelier_sabr_params[i, 4])




x_axs = np.tile(fixing_time.reshape(-1,1), strike_mat.shape[1])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x_axs, strike_mat*100, calibrated_impl_vols*10000, cmap = cm.gray, label = "SABR")
ax.plot_surface(x_axs, strike_mat*100, cap_mat,cmap = cm.viridis, label = "Market")
ax.view_init(20, 240)
ax.set_xlabel('Fixing Time')
ax.set_ylabel('Strike %')
ax.set_zlabel('Implied Volatility (bps)')
ax.set_title('Caplet Implied Volatility Surface')


#RMSE perc
np.sqrt(np.mean(((calibrated_impl_vols/(cap_mat/10000) - 1))**2))*100

# =============================================================================
# Fit instantaneous volatility functions g & H
# 2 different parametric forms available, exponential and polynomial
# =============================================================================
def instant_vol_exp(t, expiry, params):
    '''
    4 factors
    params 4x1 array of parameters
    num_factors = number of factors
    '''
    a, b, c, d = params
    tau = expiry - t
    
    g = (a + b * tau)*np.exp(- c * tau) + d
    return g


def calibrate_exp(params, target, fixing_grid):
    '''
    calibrate instantaneous volatility exponential form
    target = target volatility array nx1
    fixing grid = fixing times array nx1 for the target volatility
    '''
    vol_avgs = np.zeros(len(fixing_grid))
    
    
    for i in range(len(fixing_grid)):
        tau = fixing_grid[i]
        sol = integrate.quad(lambda x : instant_vol_exp(x, tau, params)**2 , 0, tau)
        vol_avgs[i] = np.sqrt(1/fixing_grid[i] * sol[0])
        
    residual = target - vol_avgs
    
    return(residual)

min_g = optimize.least_squares(lambda x : calibrate_exp(x, bachelier_sabr_params[:, 2], fixing_time), x0 = [0.1,0.1,0.1,0.1])
min_h = optimize.least_squares(lambda x : calibrate_exp(x, bachelier_sabr_params[:, 4], fixing_time), x0 = [0.1,0.1,0.1,0.1])


def vol_exp(t, expiry, params):
    tau = expiry - t
    vol_sq = integrate.quad(lambda x : instant_vol_exp(x, tau, params)**2 , 0, tau)[0]
    return vol_sq


s0_exp = np.zeros(len(fixing_time))
epsilon_exp  = np.zeros(len(fixing_time))
parameterized_vol_exp = np.zeros(len(fixing_time))
parameterized_volvol_exp = np.zeros(len(fixing_time))
for i in range(len(fixing_time)):
    parameterized_vol_exp [i] = np.sqrt((1/fixing_time[i]) * vol_exp(0, fixing_time[i], min_g.x))
    s0_exp[i] = bachelier_sabr_params[i, 2] / parameterized_vol_exp [i]
    parameterized_volvol_exp[i] = np.sqrt((1/fixing_time[i]) * vol_exp(0, fixing_time[i], min_h.x))
    epsilon_exp[i] = bachelier_sabr_params[i, 4] / parameterized_volvol_exp[i]




# #g plot fit
fig = plt.figure()
ax = fig.gca()
ax.plot(bachelier_sabr_params[:, 2])
ax.plot(parameterized_vol_exp)
ax.set_title('Instantaneous Volatility Function g(T-t)')
ax.set_xlabel('Fixing Time')
ax.set_ylabel('Volatility')

#h plot fit (volvol)
fig = plt.figure()
ax = fig.gca()
ax.plot(bachelier_sabr_params[:, 4])
ax.plot(parameterized_volvol_exp)
ax.set_title('Instantaneous Volvol Function H(T-t)')
ax.set_xlabel('Fixing Time')
ax.set_ylabel('Volatility')



# =============================================================================
# fit polynomials params for g & h
# =============================================================================
def instant_vol_sq_poly(t, expiry, params):
    '''
    3-order polynomial instantaneous vol (derivative of (a + bx + cx**2 + dx**3)^2 x)
    '''
    a,b,c,d = params
    tau = expiry-t
    res = (a + tau * (b + tau * (c + d * tau))) * \
            (a + tau * (3 * b + tau * (5 * c + 7 * d * tau)))
    return res


# instantaneous_vol_poly_params = np.polyfit(maturities, bachelier_sabr_params[:, 2], 3) #g
# instantaneous_volvol_poly_params = np.polyfit(maturities, bachelier_sabr_params[:, 4], 3) #h


# #g function fit (alpha)
# plt.plot(bachelier_sabr_params[:, 2])
# plt.plot(np.polyval(instantaneous_vol_poly_params, maturities))


# # h function fit (volvol)
# plt.plot(np.polyval(instantaneous_volvol_poly_params, maturities))
# plt.plot(bachelier_sabr_params[:, 4])




# s0 = bachelier_sabr_params[:, 2]/np.polyval(instantaneous_vol_poly_params, maturities)
# epsilon = bachelier_sabr_params[:, 4]/np.polyval(instantaneous_volvol_poly_params, maturities)


# # quad(lambda x : instant_vol_sq_poly(0, x, [a[0], a[1], a[2], a[3]]), 0, 3)[0]
# # 3 *(a[0] + a[1]*3 + a[2]*9 + a[3]*27)**2 


# =============================================================================
# Forward-Volatility Correlation
# get fwd_vol correlation from rho SABR params
# =============================================================================
corr_fwd_vol = np.zeros((19, 19))
corr_fwd_vol[np.diag_indices(19)] = bachelier_sabr_params[:, 3]



x = fixing_time
y = x   

X, Y = np.meshgrid(x, y)
z = corr_fwd_vol

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, z, cmap = cm.gray)
ax.view_init(20, 240)
plt.title("Forward-Vol Correlation")
ax.set_xlabel('Expiry')
ax.set_ylabel('Expiry')
ax.set_zlabel('Correlation')


