# LMM SABR
This project is an initial pass for LMM SABR interest rate modelling. There are three main files that needs to be run before simulating interest rate paths for the LMM SABR model. The code for the simulation of LMM SABR model itself is under the main_LMM_SABR notebook. The three main files needed are listed below and have been run ahead of time and their outputs saved under the parameters folder. Hence, you can simply go ahead and run the main_LMM_SABR notebook.

### Forward Rates
The first notebook that needs to be run is the forward_curve_corr.ipynb. This notebook takes a dataset of daily historical forward rates and produces the forward-forward correlation as well as the volatility-volatility correlation.


### SABR
The SABR.py contains all the necessary functions needed to fit the SABR models to the caplet implied volatility matrix.


### Caplet Fitting
The caplet_fitting.ipynb notebook takes in a caplet implied volatility matrix data and fits the SABR model to each expiry of the caplets for the different strikes. By doing so we can obtain a list of SABR parameters for the different expiring caplets. Afterwards, we can also model the instantaneous volatility function fit and intantaneous volatility-volatility function fit using the calibrated $\alpha$ and $\nu$ of the SABR parameters. Finally, the forward-volatility correlation structure block can be identified by the $\rho$ SABR parameters that were fit into the caplets market in the diagonals.
 
### Bermudan Swaption Valuation
Finally, the berm_valuation notebook shows an example on how to value Bermudan Swaptions using the simulated interest rate paths under the LSM framework. Three different methods were considered: neural-network, XGBoost and least-squares for the valuation of Bermudan swaptions under the LSM framework.
