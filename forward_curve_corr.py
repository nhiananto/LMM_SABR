# =============================================================================
# Import Historical Forward Rates
# =============================================================================
import pandas as pd
import numpy as np
from scipy import interpolate, stats, optimize
import matplotlib.pyplot as plt
from arch import arch_model
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import datetime
import matplotlib.dates as mdates

path_rates = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/data/string model/newdata/"
raw_fwd = pd.read_csv(path_rates+"forwardrates.csv")
raw_fwd["Date"] = pd.to_datetime(raw_fwd["Date"], utc = False)
raw_fwd = raw_fwd[raw_fwd["Date"] >= "11/1/2019"]


#%%
#plot
date =  mdates.date2num(np.array(raw_fwd["Date"]))

x = date.reshape(-1,1)
y = np.arange(1, 21)/2 #expiries
X, Y = np.meshgrid(x, y)
z = np.array(raw_fwd.drop(["Date", "Unnamed: 0"], axis = 1)).T

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, z, cmap = cm.viridis)
ax.set_title("Historical Forward Surface")
ax.set_ylabel("Expiry")
ax.set_zlabel("Forward Rate")
ax.set_xlabel("")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval= 60))
plt.xticks(rotation = 40,  fontsize =8)
ax.tick_params(axis = 'x', pad = -2.5)
ax.view_init(20, 300)

#%%


# =============================================================================
# calculate correlation matrix
# =============================================================================
fwd_mat = np.array(raw_fwd.drop(["Date", "Unnamed: 0"], axis = 1))


#changes in forward
fwd_mat_chg = np.apply_along_axis(lambda x: np.diff(x)/x[:-1], 0, fwd_mat)


corr_chg = np.corrcoef(fwd_mat_chg.T)
corr_fwd = np.corrcoef(fwd_mat.T)



# =============================================================================
# estimate volvol - step-wise
# =============================================================================
# divide by buckets and estimate volatility per bucket
# making sure excess kurtosis is close to 0
numbuckets = 50
upper_idx = np.arange(0, fwd_mat.shape[0]+1, step =  fwd_mat.shape[0]/numbuckets)
idx_ranges = [[int(upper_idx[i-1]), int(upper_idx[i])] for i in range(1, len(upper_idx))]

vol_ts = np.zeros((len(idx_ranges), fwd_mat.shape[1])) #stepwise
vol_kurtosis = np.zeros_like(vol_ts)
for i in range(len(idx_ranges)):
    temp_volbucket = fwd_mat[idx_ranges[i][0]:idx_ranges[i][1], :]
    ts_volbucket = np.apply_along_axis(lambda x: np.diff(x)/x[:-1], 0, temp_volbucket) #calculate mean returns
    ts_volbucket_std = (ts_volbucket)/np.std(ts_volbucket, axis = 0) #standardize
    vol_kurtosis[i, :] = stats.kurtosis(ts_volbucket, axis = 0)
    vol_ts[i, :] = np.std(ts_volbucket, axis = 0)


# check excess kurtosis fir each bucket
# plt.plot(vol_ts[:,0])
# plt.plot(vol_kurtosis)
# plt.plot(np.std(vol_ts, axis = 0))
# plt.ylim(0, 1)

volvol_corr = np.corrcoef(vol_ts.T)

# =============================================================================
# estimate volvol - garch
# =============================================================================
vol_garch_ts = np.zeros_like(fwd_mat_chg)
for i in range(fwd_mat_chg.shape[1]):
    model = arch_model(fwd_mat_chg[:, i], mean='Zero', vol='GARCH', p=2, q=2, rescale = False)
    fit= model.fit()
    vol_garch_ts[:, i] = fit.conditional_volatility

volvol_corr_garch = np.corrcoef(vol_garch_ts.T)



# =============================================================================
# parametrization with doust
# =============================================================================

def doust_corr(beta, n):
    '''
    create nxn doust correlation with beta decay exponential
    n = # of semi-annual expiries
    '''
    tau = np.arange(1, n+1)/2
    a = np.exp(- beta / np.arange(1, len(tau[:-1])+1) )
    doust = np.zeros((n, n))
    dim = doust.shape
    for i in range(doust.shape[0]):
        for j in range(doust.shape[1]):
            if i == j:
                doust[i, j] = 1
            elif i > j:
                doust[i, j] = np.prod(a[j:i])
    #reflect
    doust[np.triu_indices(dim[0], 1)] = doust.T[np.triu_indices(dim[0], 1)]
    return(doust)



def calibrated_doust_eigen(target_corr):
    '''
    calibrate beta parameter for doust correlation by optimizing over the first 4 eigen values
    '''
    n =target_corr.shape[0]
    
    #optimize over sum of first 4 eigenvalues
    sol = optimize.minimize(lambda x: 
                        np.sum((np.linalg.eigvals(target_corr)[:4] - 
                                np.linalg.eigvals(doust_corr(x, n))[:4])**2), x0 = np.array([0.1]))
        
    calibrated_beta = sol.x
    print(calibrated_beta)
    return(doust_corr(calibrated_beta, n))


def calibrated_doust_element(target_mat):
    '''
    calibrate beta parameter for doust correlation by finding least squares of each matrix entry
    '''
    
    dim_mat =target_mat.shape
    
    target_entries = target_mat.ravel()
    def func_to_minimize(beta):
        residual = target_entries - doust_corr(beta, dim_mat[0]).ravel()
        return residual
    #optimize over sum of first 4 eigenvalues
    sol = optimize.least_squares(func_to_minimize,
            x0 = 0.1)
        
    calibrated_beta = sol.x
    print(calibrated_beta)
    return(doust_corr(calibrated_beta, dim_mat[0]))


doust_fwd_fwd = calibrated_doust_element(corr_chg)
doust_vol_vol = calibrated_doust_element(volvol_corr_garch)

def plot_corr_mat(corr_mat):
    dim = corr_mat.shape
    x = np.arange(1, (dim[0]+1))/2
    y = x   

    X, Y = np.meshgrid(x, y)
    z = corr_mat

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z, cmap = cm.gray)
    ax.view_init(20, 240)



plot_corr_mat(corr_chg)
plt.title('Historical Forward-Forward Correlation')



plot_corr_mat(doust_fwd_fwd)
plt.title('Forward-Forward Doust Parametrization')



plot_corr_mat(volvol_corr_garch)
plt.title('Historical Vol-Vol Correlation')


plot_corr_mat(doust_vol_vol)
plt.title('Vol-Vol Doust Parametrization')






