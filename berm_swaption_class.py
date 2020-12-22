# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:26:17 2020

@author: nhian
"""


import numpy as np


class berm_swaptions:
    def __init__(self, rate_matrix, strike, tenor, lockout, opt_type = 'rec'):
        '''
        #rate matrix 3d array rate matrix T x T x paths
        #contains initial discount curve
        '''
        
        #store params        
        self.params = {"strike":strike,
                       "tenor":tenor,
                       "lockout":lockout,
                       "rate_matrix":rate_matrix}
        
        dim = rate_matrix.shape #dim[0] x dim[1], dim[2] = # of paths
        
        #no. exercisable steps
        num_ex_steps = (tenor*2 - lockout*2)
        
        
        #(paths x exercisable steps)
        self.intrinsic = np.zeros((dim[2], dim[1])) #intrinsic
        
        
        #create additional helper matrices
        self.index = np.zeros_like(self.intrinsic) #stopping rule
        self.exp_cv = np.zeros_like(self.intrinsic) #Q
        self.value = np.zeros_like(self.intrinsic) #max of exp_cv (Q) and intrinsic value
        
        
        #cmmf compounded monthly discount factor #paths x step
        di = np.diag_indices(dim[0])[0]
        self.cmmf = np.cumprod(rate_matrix[di, di,:], axis = 0).T
        
        #option expire worthless at the end of coupon payment date
        self.intrinsic[:, -1] = 0
         
        #option cannot be exercised at the first coupon payment (acc. for initial discount curve)
        lockout_i = int(lockout * 2)
        self.intrinsic[:, :lockout_i] = 0
        
        
        '''
        compute swap rate (par rate) on exercise dates (2nd cpn until next to last cpn)
        compare strike and swap rate and calculate intrinsic value
        exercise only after exchanging cpns due on the payment date
        
        calculate starting row indexes on exercise dates
        1st exercise on 2nd cpn: lockout*2 + 2 (acc. for initial discount curve)
        last exercise on 2nd last cpn pmt (cannot exercise on last payment date)
        '''
        start_ri = lockout_i + np.arange(0, tenor *2 - lockout*2) #also column index of possible exercise dates
        
        self.exercisable_steps = start_ri
        
        #remaining coupon pmts on exercise dates
        rem_cpn_pmt = np.arange(tenor *2, 0, -1)[lockout_i:] 
        
        #row maturity index on exercise dates
        mti = rem_cpn_pmt + start_ri - 1 
        
        
        annuity = np.zeros((len(rem_cpn_pmt), dim[2])) #exercise dates x paths
        for step in range(len(rem_cpn_pmt)) :
            annuity[step,:] = np.sum(rate_matrix[start_ri[step] : (mti+1)[step], lockout_i + step, :],
                                       axis = 0)

        #exercise dates x paths
        self.par = 2 * (1 - rate_matrix[mti, start_ri, :]) / annuity
        
        #receiver
        rec = 0.5 * np.maximum(strike - self.par, 0) * annuity
        pay = 0.5 * np.maximum(self.par - strike, 0) * annuity
        
        end_step = self.exercisable_steps[-1]
        
        if opt_type == "pay":
            pmt = pay.T
        elif opt_type == "rec":
            pmt = rec.T
        else:
            print("Choose correct option type: pay/rec")
            raise
        
        self.intrinsic[:, lockout_i: end_step + 1] = pmt
        
        
        self.value[:, end_step] = self.intrinsic[:, end_step]
        self.index[:, end_step] = np.where(self.intrinsic[:, end_step] > 0, 1, 0)
        
        
    def X(self, step):
        '''
        return basis functions for that step
        3 powers of swap value & unmatured bond prices up to and including final maturity date of the swap
        returns of dimension = paths x nvar for that step
        '''
        
        #3 powers of swap value
        swap_value = self.intrinsic[:, step].reshape(-1, 1)
        swap_value = np.repeat(swap_value, 3, axis = 1) ** np.arange(1, 4)
        
        #unmatured bond prices
        end_i = self.params["tenor"]*2
        bond_prc = self.params["rate_matrix"][step:end_i, step, :].T
        
        #combine basis
        basis = np.concatenate([swap_value, bond_prc], axis = 1)
        
        return basis, basis.shape[1]#return basis and dim
    
    
    def update(self, step, output):
        '''
        #output = prediction output for 1 step ahead #V
        #dim = (paths x 1)
        '''
        
        #discount rates for all the paths (paths x 1) for 1 step
        disc = self.params["rate_matrix"][step, step, :].reshape(-1) #paths x 1
        
        self.exp_cv[:, step] = output #Q
        
        #update value matrix
        # self.value[:,step] = np.maximum(self.intrinsic[:,step], self.exp_cv[:, step])
        self.value[:,step] = np.where(self.intrinsic[:,step] > self.exp_cv[:, step],
                                      self.intrinsic[:,step],
                                      disc * self.value[:, step + 1])
        
        #update index matrix
        rows = np.where(self.intrinsic[:, step] > self.exp_cv[:, step])[0]
        self.index[rows] = 0 #reset to 0
        self.index[rows, step] = 1


####################################


class berm_swaptions_power:
    def __init__(self, rate_matrix, strike, tenor, lockout, opt_type = "rec"):
        '''
        #rate matrix 3d array rate matrix T x T x paths
        #contains initial discount curve
        '''
        
        #store params        
        self.params = {"strike":strike,
                       "tenor":tenor,
                       "lockout":lockout,
                       "rate_matrix":rate_matrix}
        
        dim = rate_matrix.shape #dim[0] x dim[1], dim[2] = # of paths
        
        #no. exercisable steps
        num_ex_steps = (tenor*2 - lockout*2)
        
        
        #(paths x exercisable steps)
        self.intrinsic = np.zeros((dim[2], dim[1])) #intrinsic
        
        
        #create additional helper matrices
        self.index = np.zeros_like(self.intrinsic) #stopping rule
        self.exp_cv = np.zeros_like(self.intrinsic) #Q
        self.value = np.zeros_like(self.intrinsic) #max of exp_cv (Q) and intrinsic value
        
        
        #cmmf compounded monthly discount factor #paths x step
        di = np.diag_indices(dim[0])[0]
        self.cmmf = np.cumprod(rate_matrix[di, di,:], axis = 0).T
        
        #option expire worthless at the end of coupon payment date
        self.intrinsic[:, -1] = 0
         
        #option cannot be exercised at the first coupon payment (acc. for initial discount curve)
        lockout_i = int(lockout * 2)
        self.intrinsic[:, :lockout_i] = 0
        
        
        '''
        compute swap rate (par rate) on exercise dates (2nd cpn until next to last cpn)
        compare strike and swap rate and calculate intrinsic value
        exercise only after exchanging cpns due on the payment date
        
        calculate starting row indexes on exercise dates
        1st exercise on 2nd cpn: lockout*2 + 2 (acc. for initial discount curve)
        last exercise on 2nd last cpn pmt (cannot exercise on last payment date)
        '''
        start_ri = lockout_i + np.arange(0, tenor *2 - lockout*2) #also column index of possible exercise dates
        
        self.exercisable_steps = start_ri
        
        #remaining coupon pmts on exercise dates
        rem_cpn_pmt = np.arange(tenor *2, 0, -1)[lockout_i:] 
        
        #row maturity index on exercise dates
        mti = rem_cpn_pmt + start_ri - 1 
        
        
        annuity = np.zeros((len(rem_cpn_pmt), dim[2])) #exercise dates x paths
        for step in range(len(rem_cpn_pmt)) :
            annuity[step,:] = np.sum(rate_matrix[start_ri[step] : (mti+1)[step], lockout_i + step, :],
                                       axis = 0)

        #exercise dates x paths
        self.par = 2 * (1 - rate_matrix[mti, start_ri, :]) / annuity
        
        #receiver
        rec = 0.5 * np.maximum(strike - (self.par)**2, 0) * annuity
        pay = 0.5 * np.maximum((self.par)**2 - strike, 0) * annuity
        
        if opt_type == "pay":
            pmt = pay.T
        elif opt_type == "rec":
            pmt = rec.T
        else:
            print("Choose correct option type: pay/rec")
            raise
        
        end_step = self.exercisable_steps[-1]
        
        self.intrinsic[:, lockout_i: end_step + 1] = pmt
        
        self.value[:, end_step] = self.intrinsic[:, end_step]
        self.index[:, end_step] = np.where(self.intrinsic[:, end_step] > 0, 1, 0)
        
        
    def X(self, step):
        '''
        return basis functions for that step
        3 powers of swap value & unmatured bond prices up to and including final maturity date of the swap
        returns of dimension = paths x nvar for that step
        '''
        
        #3 powers of swap value
        swap_value = self.intrinsic[:, step].reshape(-1, 1)
        swap_value = np.repeat(swap_value, 3, axis = 1) ** np.arange(1, 4)
        
        #unmatured bond prices
        end_i = self.params["tenor"]*2
        bond_prc = self.params["rate_matrix"][step:end_i, step, :].T
        
        #combine basis
        basis = np.concatenate([swap_value, bond_prc], axis = 1)
        
        return basis, basis.shape[1]#return basis and dim
    
    
    def update(self, step, output):
        '''
        #output = prediction output for 1 step ahead #V
        #dim = (paths x 1)
        '''
        
        #discount rates for all the paths (paths x 1) for 1 step
        disc = self.params["rate_matrix"][step, step, :].reshape(-1) #paths x 1
        
        self.exp_cv[:, step] = output #Q
        
        #update value matrix
        # self.value[:,step] = np.maximum(self.intrinsic[:,step], self.exp_cv[:, step])
        self.value[:,step] = np.where(self.intrinsic[:,step] > self.exp_cv[:, step],
                                      self.intrinsic[:,step],
                                      disc * self.value[:, step + 1])
        
        #update index matrix
        rows = np.where(self.intrinsic[:, step] > self.exp_cv[:, step])[0]
        self.index[rows] = 0 #reset to 0
        self.index[rows, step] = 1