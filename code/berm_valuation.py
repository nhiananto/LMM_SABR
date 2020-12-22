# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:08:48 2020

@author: nhian
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import scipy
from sklearn import model_selection, metrics
import os
from functools import partial
from datetime import date
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.skopt import SkOptSearch
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pickle
import time

module_path = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/code/"
os.chdir(module_path)

path_rates = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/data/lmm sabr/"
path_results = "C:\\Users\\nhian\\Dropbox\\UCLA MFE\\Spring 2020\\AFP\\run results\\"


from berm_swaption_class import *

# =============================================================================
# define custom neural net class for hyper opt tuning
# =============================================================================
class net(nn.Module):
    def __init__(self, input_dim, neurons, hidden_layers, activation_fn):
        super(net, self).__init__()
        self.activation_fn = activation_fn
        self.input_dim = input_dim
        
        self.layers= nn.ModuleList()
        #add first layer
        self.layers.append(nn.Linear(input_dim, neurons))
        
        for k in range(1, hidden_layers+1):
            self.layers.append(nn.Linear(neurons, neurons))
            
        #output layer
        self.layers.append(nn.Linear(neurons, 1))
        

    def forward(self, x):
        for i in range(len(self.layers)-1): #except last layer
            x = self.layers[i](x)
            x = self.activation_fn(x)

        #output layer (no activation)
        x = self.layers[-1](x)
        return x


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


def laguerre_scipy(x, k):
    #x: paths x nvars
    #return paths x (nvars * polynomials)
    
    dim = x.shape
    L = np.zeros((dim[0], 1)) #paths x 1 #placeholder
    
    for i in range(dim[1]):
        weight = np.exp(-x[:, i]/2)
        for j in range(k):
            L = np.concatenate([L,
                            (weight.reshape(-1,1) * 
                            scipy.special.eval_genlaguerre(j, 0, x[:,i]).reshape(-1,1))], axis = 1)
    
    L = L[:,1:]
    return(L)


# =============================================================================
# define training function with early stopping
# =============================================================================

def save_checkpoint(model, optimizer, path, filename, **kwargs):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, path + filename)
    

def load_checkpoint(model, optimizer, path, filename):
    checkpoint = torch.load(path + filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return(checkpoint)
#%%
def trainNN(config, x, y, input_size, max_num_epochs, use_tune = True,
            checkpoint_dir = "C:\\Users\\nhian\\Desktop\\ray_results_12_13\\"):
    '''
    config = dict of number of neurons, num hidden, activation function used, learning rate, batch size
    '''
    #n_epochs = 100, verbose = True, checkpoint_dir = "C:/Users/nhian/Desktop/checkpoints/"
    
    
    model = net(input_size, config["num_neurons"], config["num_hidden"], config["activation_fn"])
    
    # patience = 6 #early stopping
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    criterion = torch.nn.functional.mse_loss
    optimizer=optim.Adam(model.parameters(),lr= config["lr"])
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'min',
                                     patience = 10, threshold_mode = 'rel', threshold = 1e-4)
    
    
    x = x.reshape(-1, input_size)
    y = y.reshape(-1,1)
    
    #split
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,
                                                                    y, test_size=0.1) 
    
    # x = torch.tensor(x, dtype = torch.float).to(device)
    # y = torch.tensor(y, dtype = torch.float).to(device)
    
    x_train = torch.tensor(x_train,dtype=torch.float)
    y_train = torch.tensor(y_train,dtype=torch.float)
    x_test = torch.tensor(x_test,dtype=torch.float)
    y_test = torch.tensor(y_test,dtype=torch.float)
    
    
    train_dataset = torch.utils.data.TensorDataset(x_train.view(-1, input_size), y_train.view(-1,1))
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size= 32,  #
                                               shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test.view(-1, input_size), y_test.view(-1,1))
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                               batch_size= 32,  #
                                               shuffle=True)


    #patience ct for early stopping criteria
    # patience_ct = 0
    
    #track losses
    train_losses, test_losses = [], []
    test_r2 = []
    avg_train_losses, avg_test_losses  = [], []
    avg_test_r2  = []
    
    #set initial losses as inf
    avg_train_losses.append(float('inf'))
    avg_test_losses.append(float('inf'))
    
    def accuracy_r2(y_true, y_pred):
        #for tensors
        y_true = y_true.detach().numpy().reshape(-1,1)
        y_pred = y_pred.detach().numpy().reshape(-1,1)
        return metrics.r2_score(y_true, y_pred)
    
    for epoch in range(max_num_epochs):
        # =======================
        # train        
        # =======================
        model.train() #training mode    
        for i, (features, price) in enumerate(train_loader):
            optimizer.zero_grad()
            features, price= features.to(device), price.to(device)
            
            #forward
            model_output = model(features)
            
            #backprop
            loss = criterion(model_output,price)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
            
        # ======================
        # eval
        # ======================
        model.eval() # prep model for evaluation
        for i, (features, price) in enumerate(test_loader):
            with torch.no_grad():
                features = features.to(device)
                price = price.to(device)
                # forward pass
                model_output = model(features)
                
                # calculate the loss
                loss = criterion(model_output, price)
                # record validation loss
                test_losses.append(loss.item())
        

        
        train_loss, test_loss = np.mean(train_losses), np.mean(test_losses)
        
        
        
        #whole validation set
        test_r2 = accuracy_r2(test_dataset.tensors[1], model(test_dataset.tensors[0]))
    
        avg_train_losses.append(train_loss)
        avg_test_losses.append(test_loss)
        avg_test_r2.append(test_r2)
        
        #lr scheduler min mse
        lr_scheduler.step(test_loss)
        
        
        #save chkpoint and report to tune
        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint\\")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)
        if use_tune:
            print("Save chk")
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                  path = os.path.join(checkpoint_dir, "checkpoint")
                  torch.save((model.state_dict(), optimizer.state_dict()), path)
                  
            tune.report(loss= test_loss, accuracy= test_r2) #report back to tune
            
            
            
        #clear list
        train_losses = []
        test_losses = []
        


        #print
        # if verbose:
        #       print_msg = "[{0}/{1}] train_loss = {2:.5f}; valid_loss = {3:.5f}; train_r2 = {4:3f}; \
        #                       test_r2 = {5:3f}".\
        #           format(epoch, n_epochs, train_loss, test_loss, train_r2, test_r2)
        #       print(print_msg)
              
        # #early stopping        
        # if (avg_test_losses[epoch+1] - avg_test_losses[epoch]) > 0: #validation loss does not decrease
        #     patience_ct += 1 #add 1 to patience counter
        #     if verbose:
        #         print("Validation Loss did not decrease, Patience:{}".format(patience_ct))
        # else:
        #     # save_checkpoint(model, optimizer, checkpoint_dir, "chk.tar")
        #     if verbose:
        #         print("Validation Loss decreased {:.5f} --> {:.5f}, saving checkpoint...".\
        #               format(avg_test_losses[epoch],avg_test_losses[epoch+1]))
            
        # if patience_ct > patience:
        #     if verbose:
        #         print("Early Stopping")
        #     break
        
    
    # load the last checkpoint with the best model
    # load_checkpoint(model, optimizer, checkpoint_dir, "chk.tar")
    # y_hat = model(x.view(-1, input_size)).detach().numpy().reshape(-1,1)

    print("Finished Training")    




# =============================================================================
# setup environament for ray tune
# =============================================================================
'''
config: dictionary of search space
num_neurons = #of neurons for each hidden layer
num_hidden = number of hidden layers (not incl. output layer)
activation_fn = activation function for each hidden layer
batch size = batch size for mini batch GD
lr = learning rate for adam
'''

#%%
def main_NN(data, num_samples=15, max_num_epochs=30,
            metric = 'loss', mode = 'min',
            checkpoint_dir = "C:\\Users\\nhian\\Desktop\\ray_results_12_13\\",
            tune_path = "C:/Users/nhian/Dropbox/UCLA MFE/Spring 2020/AFP/ray_results/",
            experiment_name = "experiment"+str(date.today()),
            trial_name = None):
    '''
    data = tuple of (X, y) dataset used for training and validation
    num_samples = samples to search from search space
    max_num_epochs = max number of epochs to train the NN
    
    search over NN hyperspace defined by config
    max_num_epochs = max epochs for ASHAScheduler to terminate training
    num_samples = num trials
    
    trial_name= current trial name
    trial_dir = same as trial name
    
    '''
    
    # config = {
    #     "num_neurons": tune.choice([16, 32, 64, 128]),
    #     "num_hidden": tune.choice([2,3,4]),
    #     "activation_fn" : tune.choice([F.relu, F.leaky_relu]),
    #     "batch_size": tune.choice([32]),
    #     "lr": tune.loguniform(1e-4, 1e-1)
    # }
    
    bayes_searchspace = {
        "num_neurons": Categorical([int(x) for x in  2**np.arange(4,8)]),
        "num_hidden": Integer(2, 4, 'uniform'),
        "activation_fn" : Categorical([F.relu]),
        # "batch_size": Integer(16, 32),#Categorical([int(16), int(32)]),
        "lr": Real(1e-4, 1e-2, 'log-uniform')
        }
    
    # bayesopt = BayesOptSearch(metric="accuracy", mode="max")
    skopt_search = SkOptSearch(space = bayes_searchspace, metric="accuracy", mode="max")
    
    
    '''
    can set metric/mode in scheduler or tune.run
    ASHA scheduler can set max_num_epochs
    grace_period = min number of iterations before stopping
    '''
    scheduler = ASHAScheduler(
                    metric= metric, #loss
                    mode= mode, #min
                    time_attr='training_iteration',
                    max_t=max_num_epochs,
                    grace_period=10,
                    reduction_factor=2)
    
    '''
    CLIReporter for python console
    what to print to console
    '''
    reporter = CLIReporter( 
        parameter_columns=["lr", "num_neurons", "num_hidden"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    #get dataset
    x, y = data
    input_size = x.shape[1]
    

    #need to register function (trainable) (or use partial and fill out other arguments)
    # tune.register_trainable("fc_nn",
    #                         lambda cfg : trainNN(config = cfg, x = x, y = y,
    #                                  input_size = input_size, max_num_epochs = max_num_epochs))
    
    def trial_name_string(trial):
        return trial_name+str(trial.trial_id)
    
    
    result = tune.run(
            partial(trainNN, x = x, y = y, input_size = input_size, max_num_epochs = max_num_epochs,
                    checkpoint_dir = checkpoint_dir),
            resources_per_trial={"cpu": 3, "gpu": 0},
            # config=config,
            search_alg = skopt_search,
            num_samples=num_samples,
            scheduler=scheduler,
            reuse_actors = True,
            progress_reporter=reporter,
            name= experiment_name,
            local_dir = tune_path,
            trial_name_creator = trial_name_string,
            trial_dirname_creator = trial_name_string)
    
    #get best trial
    best_trial = result.get_best_trial(metric, mode, "last") #"accuracy" , max
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    best_trained_model = net(input_size, best_trial.config["num_neurons"],
                             best_trial.config["num_hidden"], best_trial.config["activation_fn"])
    
    
    # best_checkpoint_dir = best_trial.checkpoint.value
    
    # model_state, optimizer_state = torch.load(os.path.join(
    #     checkpoint_dir, "checkpoint"))
    
    
    # best_trained_model.load_state_dict(model_state)

    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    
    
    return best_trained_model, result
#%%
def retrainNN(x, y, config, max_num_epochs):
    '''
    config = dict of number of neurons, num hidden, activation function used, learning rate, batch size
    '''
    #n_epochs = 100, verbose = True, checkpoint_dir = "C:/Users/nhian/Desktop/checkpoints/"
    
    input_size = x.shape[1]
    
    model = net(input_size, config["num_neurons"], config["num_hidden"], config["activation_fn"])
    
    # patience = 6 #early stopping
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    
    criterion = torch.nn.functional.mse_loss
    optimizer=optim.Adam(model.parameters(),lr= config["lr"])
    
    lr_scheduler = ReduceLROnPlateau(optimizer, mode = 'min',
                                     patience = 10, threshold_mode = 'rel', threshold = 1e-4)
    
    
    
    x = x.reshape(-1, input_size)
    y = y.reshape(-1,1)
    
    #split
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,
                                                                    y, test_size=0.1) 
    
    # x = torch.tensor(x, dtype = torch.float).to(device)
    # y = torch.tensor(y, dtype = torch.float).to(device)
    
    x_train = torch.tensor(x_train,dtype=torch.float)
    y_train = torch.tensor(y_train,dtype=torch.float)
    x_test = torch.tensor(x_test,dtype=torch.float)
    y_test = torch.tensor(y_test,dtype=torch.float)
    
    
    train_dataset = torch.utils.data.TensorDataset(x_train.view(-1, input_size), y_train.view(-1,1))
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                               batch_size= 32,  #
                                               shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test.view(-1, input_size), y_test.view(-1,1))
    
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                               batch_size= 32,  #
                                               shuffle=True)


    #patience ct for early stopping criteria
    # patience_ct = 0
    
    #track losses
    train_losses, test_losses = [], []
    test_r2 = []
    avg_train_losses, avg_test_losses  = [], []
    avg_test_r2  = []
    
    #set initial losses as inf
    avg_train_losses.append(float('inf'))
    avg_test_losses.append(float('inf'))
    
    def accuracy_r2(y_true, y_pred):
        #for tensors
        y_true = y_true.detach().numpy().reshape(-1,1)
        y_pred = y_pred.detach().numpy().reshape(-1,1)
        return metrics.r2_score(y_true, y_pred)
    
    for epoch in range(max_num_epochs):
        print(epoch)
        # =======================
        # train        
        # =======================
        model.train() #training mode    
        for i, (features, price) in enumerate(train_loader):
            optimizer.zero_grad()
            features, price= features.to(device), price.to(device)
            
            #forward
            model_output = model(features)
            
            #backprop
            loss = criterion(model_output,price)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
            
        # ======================
        # eval
        # ======================
        model.eval() # prep model for evaluation
        for i, (features, price) in enumerate(test_loader):
            with torch.no_grad():
                features = features.to(device)
                price = price.to(device)
                # forward pass
                model_output = model(features)
                
                # calculate the loss
                loss = criterion(model_output, price)
                # record validation loss
                test_losses.append(loss.item())
        

        
        train_loss, test_loss = np.mean(train_losses), np.mean(test_losses)
        
        
        
        #whole validation set
        test_r2 = accuracy_r2(test_dataset.tensors[1], model(test_dataset.tensors[0]))
    
        avg_train_losses.append(train_loss)
        avg_test_losses.append(test_loss)
        avg_test_r2.append(test_r2)
        
        #lr scheduler min mse
        lr_scheduler.step(test_loss)
        
        
    return model, model.state_dict(), optimizer.state_dict()



#%%
'''
sample dataset LMM SABR
'''

def forwards_to_zeros(forwards, spot_fwd):
    '''
    forwards = vector of semiannual forward rates with 0.5 spaced grid 
                with starting fixing date 0.5
    spot_fwd = 6mth spot fwd (short-rate)
    '''
    disc_bonds = np.array((1+spot_fwd/2)**(-1))
    zeros = (1 + forwards/2)**(-1)*disc_bonds
    zeros = np.hstack((disc_bonds, zeros))
    
    return(zeros)

def forward_mat_to_zeros(forward_mat):
    simulated_zeros_full = np.zeros_like(forward_mat)
    for j in range(forward_mat.shape[1]):
        for k in range(forward_mat.shape[2]):
            simulated_zeros_full[j:, j, k] = forwards_to_zeros(forward_mat[(j+1):, j, k],
                                                              forward_mat[j, j, k])
    
    return(simulated_zeros_full)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(obj, filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)



with open(path_rates+'simulated_fwd_full_10k.pkl', 'rb') as handle:
    sim_fwd_10k  = pickle.load(handle)

sim_rates_10k = forward_mat_to_zeros(sim_fwd_10k)

del sim_fwd_10k



with open(path_rates+'simulated_fwd_full_50k.pkl', 'rb') as handle:
    sim_fwd_50k  = pickle.load(handle)

sim_rates_50k = forward_mat_to_zeros(sim_fwd_50k)

del sim_fwd_50k

with open(path_rates+'simulated_fwd_full_100k.pkl', 'rb') as handle:
    sim_fwd_100k  = pickle.load(handle)

sim_rates_100k = forward_mat_to_zeros(sim_fwd_100k)

del sim_fwd_100k


#%%

def zeros_to_forward_par(zeros, expiry, tenor):
    '''
    convert zeros array to forward par rate at the given expiry and tenor
    under semi-annual maturing at [0.5, 1, 1.5, 2 ...]
    
    expiry: in years
    tenor: in years
    '''
    
    ri = int(2*expiry) - 1
    
    fpar = 2 * (zeros[ri] - zeros[ri + int(tenor*2)]) / np.sum(zeros[ri+1: ri + 1 + int(2*tenor)])
    return fpar



#%%
'''
=======================================================================
Bermudan Swaption Valuations with Neural Network
Set up environment and train hyperparameters for nn using config
Simulate interest rate model
loop through exercise steps to calculate continuation value
=======================================================================
'''

def oos_nn(berm_swaptions, model, step, laguerre = False):
    '''
    berm_swaptions = OOS bermudan swaptions class
    model = neural net model
    '''
    itm = np.where(berm_swaptions.intrinsic[:, step] > 0)[0]
    
    X = berm_swaptions.X(step)[0]
    initial_dim = X.shape
    
    X = X[itm, :]
    
    if laguerre:
        X = laguerre_scipy(X, 3)
    
    y = berm_swaptions.value[:, step+1]
    y = berm_swaptions.params["rate_matrix"][step, step, itm] * y[itm] #discount to now
    
    pred = model(torch.tensor(X, dtype = torch.float)).detach().numpy().reshape(-1,1)
    
    pred_all = np.zeros((initial_dim[0], 1))
    pred_all[itm, :] = pred
    
    #update
    berm_swaptions.update(step, pred_all.ravel())
    
    #oos r2
    r2_score = metrics.r2_score(pred.reshape(-1,1), y.reshape(-1,1))
    return(r2_score)
#%%

def berm_nn(berm_swaptions, sim_rates, strike, lockout, tenor, opt_type = "rec", max_num_epochs = 30,
                num_samples = 10, metric = 'loss', mode = 'min',
                test_size = 0.1, seed = 123):
    '''
    berm_swaptions: class
    sim_zeros : simulated zeros matrix n x n x paths
    strike: strike for the swaption
    lockout: in years
    tenor: in years
    test_size: float to split the dataset to training-testing split
    seed: seed for splitting
    '''
    total_dim = sim_rates.shape
    total_paths = total_dim[2] #no paths
    
    testing_paths = int(test_size * total_paths)
    training_paths = total_paths - testing_paths
    
    np.random.seed(123)
    testing_idx= np.sort(np.random.choice(np.arange(total_paths), testing_paths ,replace = False))
    training_idx = np.setdiff1d(np.arange(total_paths), testing_idx)
    
    sim = berm_swaptions(sim_rates[:, :, training_idx],
          strike = strike, tenor = tenor, lockout = lockout, opt_type = opt_type)
    
    #out of sample
    sim_oos = berm_swaptions(sim_rates[:, :, testing_idx],
                             strike = strike, tenor = tenor, lockout = lockout, opt_type = opt_type)
    
    steps = sim.exercisable_steps
    steps = np.flip(steps)[1:]
      
    tune_result_list = []
    model_list = []
    
    best_results_r2 = {"Step":[], "Accuracy":[]}
    oos_r2 = {"Step":[], "Accuracy":[]}
    
    sampling_increase = 0#np.cumsum(steps%3)
    total_time = 0
    ct = 0
    for i in steps:
        '''
        #at each time step t_{M-1} compute V_{t_{M-1}} = max(h_{tm-1}, Q_{tm-1})
        #where Q is the expected continuation value
        #V will be used as an output for training the NN
        
        #first calculate Q at tm-1
        #which is the discounted value of G at tm
        #G is the approximated function from the neural network    
        '''
        itm = np.where(sim.intrinsic[:, i] > 0)[0]
        
        X = sim.X(i)[0]
        X = X[itm, :]
        
        X = laguerre_scipy(X, 3)
        
        y = sim.value[:, i+1]
        y = sim.params["rate_matrix"][i, i, itm] * y[itm] #discount to now
        
        t0 = time.time()
        #===================================================================================
        # NN model
    
   
        # _, tune_result = main_NN((X, y),
        #                             num_samples = int(num_samples + 0),
        #                             max_num_epochs = max_num_epochs,
        #                    tune_path = "C:/Users/nhian/Desktop/ray_results_12_08/",
        #                    experiment_name = "nn_step_"+str(i), trial_name = "nn",
        #                    metric = metric, mode = mode)
        
        config = {"num_neurons":32,
                  "num_hidden":3,
                  "activation_fn":F.relu,
                  "lr": 0.0003
                      }#tune_result.get_best_trial('accuracy', 'max', 'last').config
        
        model, model_state, optim_state =  retrainNN(X, y, config, max_num_epochs)
        
        
        # tune_result_list.append(tune_result)
        model_list.append(model)
        

        #===================================================================================
        t1 = time.time()
        total_time += (t1 - t0)
        
        pred = model(torch.tensor(X, dtype =  torch.float)).detach().numpy().reshape(-1,1)
        pred_all = np.zeros((training_paths , 1))
        pred_all[itm, :] = pred
        


        best_results_r2["Step"].append(i)
        best_results_r2["Accuracy"].append(metrics.r2_score(pred.reshape(-1,1), y.reshape(-1,1)))
        
        sim.update(i, pred_all.ravel())
        
        print("Step: " + str(i))
        
        # =============================================================================
        # OOS Test
        # =============================================================================
        oos_r2["Step"].append(i)
        
        
        oos_r2["Accuracy"].append(oos_nn(berm_swaptions = sim_oos,
                                           model = model,
                                           step = i, laguerre = True)) #append r2 and update
        ct += 1
        
    
    results_table = pd.DataFrame({"Step": steps,
                                 "Training R2": best_results_r2["Accuracy"],
                                 "OOS R2": oos_r2["Accuracy"],
                                  "Exc Pr": np.sum(sim.index[:, steps], axis = 0)/training_paths,
                                  "Exc Pr OOS": np.sum(sim_oos.index[:, steps], axis = 0)/testing_paths},
                                 )
    print(results_table)
    
    price = np.sum(sim.cmmf[:, :-1] * np.multiply(sim.index[:,1:], sim.value[:,1:]))/training_paths 
    print(price)
    
    price_oos = np.sum(sim_oos.cmmf[:, :-1] * np.multiply(sim_oos.index[:,1:], sim_oos.value[:,1:]))/testing_paths
    print(price_oos)
    
    return (price, price_oos, results_table, total_time)


        
        
        
#%%

nn_results_1_10 = []
nn_results_1_5 = []


#1 NC 10
nn_results_1_10.append(berm_nn(berm_swaptions, sim_rates_10k, strike = 0.003, lockout = 1,
                               tenor = 10, opt_type = "rec",
                               num_samples = 3, max_num_epochs = 25, test_size = 0.1, seed = 123))

save_object(nn_results_1_10, "nn_results_1_10.pkl")

nn_results_1_10.append(berm_nn(berm_swaptions, sim_rates_50k, strike = 0.003, lockout = 1,
                               tenor = 10, opt_type = "rec",
                               num_samples = 3, max_num_epochs = 25, test_size = 0.1, seed = 123))

save_object(nn_results_1_10, "nn_results_1_10.pkl")

nn_results_1_10 .append(berm_nn(berm_swaptions, sim_rates_100k, strike = 0.003, lockout = 1,
                               tenor = 10, opt_type = "rec",
                               num_samples = 3, max_num_epochs = 25, test_size = 0.1, seed = 123))

save_object(nn_results_1_10, "nn_results_1_10.pkl")



#1 NC 5

# nn_results_1_5.append(berm_xgb(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_50k, strike = 0.0025, lockout = 1, tenor = 5, search_iter = 5, cv = 5))


# nn_results_1_5.append(berm_xgb(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_100k, strike = 0.0025, lockout = 1, tenor = 5, search_iter = 5, cv = 5))


# =============================================================================
# SD
# =============================================================================
nn_1_10_sd = []

for i in range(10):
    sample_sim_rates_20k = sim_rates_100k[:,:,
                        np.random.choice(np.arange(100000), 20000, False)]
    nn_1_10_sd.append(berm_nn(berm_swaptions, sample_sim_rates_20k, strike = 0.003, lockout = 1,
                                   tenor = 10, opt_type = "rec",
                                   num_samples = 3, max_num_epochs = 25,
                                   test_size = 0.1, seed = 123))

# save_object(nn_1_10_sd, path_results+"nn_1_10_sd")


#%%
# =============================================================================
# power
# =============================================================================
nn_results_1_10_power = []
nn_results_1_5_power = []


#1 NC 10


nn_results_1_10_power.append(berm_nn(berm_swaptions_power, sim_rates_50k, 0.003, 1, 10, opt_type = "rec", num_samples = 3,
                test_size = 0.1, seed = 123))

save_object(nn_results_1_10_power, "nn_results_1_10_power.pkl")

nn_results_1_10_power.append(berm_nn(berm_swaptions_power, sim_rates_50k, 0.003, 1, 10, opt_type = "rec", num_samples = 3,
                test_size = 0.1, seed = 123))


save_object(nn_results_1_10_power, "nn_results_1_10_power.pkl")




#1 NC 5
# nn_results_1_5_power.append(berm_nn(berm_swaptions_power, sim_rates_50k, 0.0025, 1, 5, opt_type = "rec", num_samples = 3,
#                 test_size = 0.1, seed = 123))


# nn_results_1_5_power.append(berm_nn(berm_swaptions_power, sim_rates_50k, 0.0025, 1, 5, opt_type = "rec", num_samples = 3,
#                 test_size = 0.1, seed = 123))









#%%
def oos_pred(berm_swaptions, model, step, laguerre = False):
    '''
    berm_swaptions = OOS bermudan swaptions class
    model = out model with predict method (X,y)
    '''
    itm = np.where(berm_swaptions.intrinsic[:, step] > 0)[0]
    
    X = berm_swaptions.X(step)[0]
    initial_dim = X.shape
    
    X = X[itm, :]
    
    if laguerre:
        X = laguerre_scipy(X, 3)
    
    y = berm_swaptions.value[:, step+1]
    y = berm_swaptions.params["rate_matrix"][step, step, itm] * y[itm] #discount to now
    
    pred = model.predict(X).reshape(-1,1)
    
    pred_all = np.zeros((initial_dim[0], 1))
    pred_all[itm, :] = pred
    
    #update
    berm_swaptions.update(step, pred_all.ravel())
    
    #oos r2
    r2_score = metrics.r2_score(pred.reshape(-1,1), y.reshape(-1,1))
    return(r2_score)

#%%
'''
=======================================================================
Bermudan Swaption Valuations with Linear Regression
Baseline Results
=======================================================================
'''
def berm_lr(berm_swaptions, sim_rates, strike, lockout, tenor, opt_type = "rec",
                test_size = 0.1, seed = 123):
    '''
    berm_swaptions: class
    sim_zeros : simulated zeros matrix n x n x paths
    strike: strike for the swaption
    lockout: in years
    tenor: in years
    test_size: float to split the dataset to training-testing split
    seed: seed for splitting
    '''
    total_dim = sim_rates.shape
    total_paths = total_dim[2] #no paths
    
    testing_paths = int(test_size * total_paths)
    training_paths = total_paths - testing_paths
    
    np.random.seed(123)
    testing_idx= np.sort(np.random.choice(np.arange(total_paths), testing_paths ,replace = False))
    training_idx = np.setdiff1d(np.arange(total_paths), testing_idx)
    
    sim = berm_swaptions(sim_rates[:, :, training_idx],
          strike = strike, tenor = tenor, lockout = lockout, opt_type = opt_type)
    
    #out of sample
    sim_oos = berm_swaptions(sim_rates[:, :, testing_idx],
                             strike = strike, tenor = tenor, lockout = lockout, opt_type = opt_type)
    
    steps = sim.exercisable_steps
    steps = np.flip(steps)[1:]
      
    
    best_results_r2 = {"Step":[], "Accuracy":[]}
    oos_r2 = {"Step":[], "Accuracy":[]}
    
    total_time = 0
    for i in steps:
        '''
        #at each time step t_{M-1} compute V_{t_{M-1}} = max(h_{tm-1}, Q_{tm-1})
        #where Q is the expected continuation value
        #V will be used as an output for training the NN
        
        #first calculate Q at tm-1
        #which is the discounted value of G at tm
        #G is the approximated function from the neural network    
        '''
        itm = np.where(sim.intrinsic[:, i] > 0)[0]
        
        X = sim.X(i)[0]
        X = X[itm, :]
        
        X = laguerre_scipy(X, 3)
        
        y = sim.value[:, i+1]
        y = sim.params["rate_matrix"][i, i, itm] * y[itm] #discount to now
        
        t0 = time.time()
        #===================================================================================
        # model
        
        out = LinearRegression().fit(X, y.reshape(-1,1))
        pred= out.predict(X).reshape(-1,1)
        
        pred_all = np.zeros((training_paths , 1))
        pred_all[itm, :] = pred
        
        #===================================================================================
        t1 = time.time()
        total_time += (t1 - t0)
        
        
        best_results_r2["Step"].append(i)
        best_results_r2["Accuracy"].append(metrics.r2_score(out.predict(X).reshape(-1,1), y.reshape(-1,1)))
        
        sim.update(i, pred_all.ravel())
        
        print("Step: " + str(i))
        
        # =============================================================================
        # OOS Test
        # =============================================================================
        oos_r2["Step"].append(i)
        oos_r2["Accuracy"].append(oos_pred(berm_swaptions = sim_oos,
                                           model = out,
                                           step = i, laguerre = True)) #append r2 and update
        
    
    results_table = pd.DataFrame({"Step": steps,
                                 "Training R2": best_results_r2["Accuracy"],
                                 "OOS R2": oos_r2["Accuracy"],
                                  "Exc Pr": np.sum(sim.index[:, steps], axis = 0)/training_paths,
                                  "Exc Pr OOS": np.sum(sim_oos.index[:, steps], axis = 0)/testing_paths},
                                 )
    print(results_table)
    
    price = np.sum(sim.cmmf[:, :-1] * np.multiply(sim.index[:,1:], sim.value[:,1:]))/training_paths 
    print(price)
    
    price_oos = np.sum(sim_oos.cmmf[:, :-1] * np.multiply(sim_oos.index[:,1:], sim_oos.value[:,1:]))/testing_paths
    print(price_oos)
    
    return (price, price_oos, results_table, total_time)
  

#%%

# fatm_10 = 0.00998704745020704
# fatm_5 = 0.005275299692111061


lr_results_1_10 = []
lr_results_1_5 = []


lr_results_1_10.append(berm_lr(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_10k, strike = 0.003, lockout = 1, tenor = 10))

lr_results_1_10.append(berm_lr(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_50k, strike = 0.003, lockout = 1, tenor = 10))

lr_results_1_10.append(berm_lr(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_100k, strike = 0.003, lockout = 1, tenor = 10))


save_object(lr_results_1_10, path_results+"lr_results_1_10.pkl")




# lr_results_1_5.append(berm_lr(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_50k, strike = 0.0025, lockout = 1, tenor = 5))

# lr_results_1_5.append(berm_lr(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_100k, strike = 0.0025, lockout = 1, tenor = 5))



# =============================================================================
# Power option
# =============================================================================

lr_results_1_10_power = []



lr_results_1_10_power.append(berm_lr(berm_swaptions = berm_swaptions_power,
                          sim_rates = sim_rates_10k, strike = 0.0003, lockout = 1, tenor = 10))

lr_results_1_10_power.append(berm_lr(berm_swaptions = berm_swaptions_power,
                          sim_rates = sim_rates_50k, strike = 0.001, lockout = 1, tenor = 10))

lr_results_1_10_power.append(berm_lr(berm_swaptions = berm_swaptions_power,
                          sim_rates = sim_rates_100k, strike = 0.001, lockout = 1, tenor = 10))


save_object(lr_results_1_10_power, path_results+"lr_results_1_10_power.pkl")


lr_results_1_5_power = []

# lr_results_1_5_power.append(berm_lr(berm_swaptions = berm_swaptions_power,
#                           sim_rates = sim_rates_50k, strike = 0.0025, lockout = 1, tenor = 5))

# lr_results_1_5_power.append(berm_lr(berm_swaptions = berm_swaptions_power,
#                           sim_rates = sim_rates_100k, strike = 0.0025, lockout = 1, tenor = 5))



# =============================================================================
# SD
# =============================================================================

lr_1_10_sd = []

for i in range(10):
    sample_sim_rates_20k = sim_rates_100k[:,:,
                        np.random.choice(np.arange(100000), 20000, False)]
    lr_1_10_sd.append(berm_lr(berm_swaptions = berm_swaptions,
                          sim_rates = sample_sim_rates_20k, strike = 0.003, lockout = 1, tenor = 10))

save_object(lr_1_10_sd, path_results+"lr_1_10_sd.pkl")


#%%
'''
=======================================================================
Bermudan Swaption Valuations with XGBoost
Bayesian Opt using skopt
=======================================================================
'''
def berm_xgb(berm_swaptions, sim_rates, strike, lockout, tenor, test_size = 0.1,
             search_iter = 15, cv = 3, search_scoring = 'neg_mean_squared_error',
             seed = 123):
    '''
    berm_swaptions: class
    sim_zeros : simulated zeros matrix n x n x paths
    strike: strike for the swaption
    lockout: in years
    tenor: in years
    test_size: float to split the dataset to training-testing split
    seed: seed for splitting
    '''
    total_dim = sim_rates.shape
    total_paths = total_dim[2] #no paths
    
    testing_paths = int(test_size * total_paths)
    training_paths = total_paths - testing_paths
    
    np.random.seed(123)
    testing_idx= np.sort(np.random.choice(np.arange(total_paths), testing_paths ,replace = False))
    training_idx = np.setdiff1d(np.arange(total_paths), testing_idx)
    
    sim = berm_swaptions(sim_rates[:, :, training_idx],
          strike = strike, tenor = tenor, lockout = lockout)
    
    #out of sample
    sim_oos = berm_swaptions(sim_rates[:, :, testing_idx],
                             strike = strike, tenor = tenor, lockout = lockout)
    
    steps = sim.exercisable_steps
    steps = np.flip(steps)[1:]
     
    model_list = []
    
    best_results_r2 = {"Step":[], "Accuracy":[]}
    oos_r2 = {"Step":[], "Accuracy":[]}
    
    total_time = 0
    for i in steps:
        '''
        #at each time step t_{M-1} compute V_{t_{M-1}} = max(h_{tm-1}, Q_{tm-1})
        #where Q is the expected continuation value
        #V will be used as an output for training the NN
        
        #first calculate Q at tm-1
        #which is the discounted value of G at tm
        #G is the approximated function from the neural network    
        '''
        itm = np.where(sim.intrinsic[:, i] > 0)[0]
        
        X = sim.X(i)[0]
        X = X[itm, :]
        
        X = laguerre_scipy(X, 3)
        
        y = sim.value[:, i+1]
        y = sim.params["rate_matrix"][i, i, itm] * y[itm] #discount to now
        
        t0 = time.time()
        #===================================================================================
        # model
        
        model = XGBRegressor()
        
        param_test = {
                'learning_rate': Real(0.01, 0.75, 'log-uniform'),
                'min_child_weight': Integer(0, 10, 'uniform'),
                'max_depth': Integer(3, 35, 'uniform'),
                'max_delta_step': Integer(0, 20),
                'subsample': Real(0.1, 1.0, 'uniform'),
                'colsample_bytree': Real(0.01, 1.0, 'uniform'),
                'colsample_bylevel': Real(0.01, 1.0, 'uniform'),
                'reg_lambda': Real(1e-9, 10, 'log-uniform'),
                'reg_alpha': Real(1e-9, 1e-2, 'log-uniform'),
                'gamma': Real(1e-9, 1e-3, 'log-uniform'), # minsplit loss
                'n_estimators': Integer(125, 350)
                }
        
        
        gsearch = BayesSearchCV(estimator = model, n_iter = search_iter,
                              search_spaces= param_test,
                              scoring= search_scoring, cv=cv, refit = True, random_state = seed)
        
        
        search_res = gsearch.fit(X, y)
        out = gsearch.best_estimator_#XGBRegressor(**gsearch.best_params_).fit(X,y)
        
        model_list.append(gsearch.best_params_)
        
        pred = out.predict(X).reshape(-1,1)
        pred_all = np.zeros((training_paths, 1))
        pred_all[itm, :] = pred
        
        #===================================================================================
        t1 = time.time()
        total_time += t1 - t0
        
        best_results_r2["Step"].append(i)
        best_results_r2["Accuracy"].append(metrics.r2_score(out.predict(X).reshape(-1,1), y.reshape(-1,1)))
        
        #update
        sim.update(i, pred_all.ravel())
        
        print("Step: " + str(i))
        print(best_results_r2["Accuracy"][-1])
        
        
        # =============================================================================
        # OOS Test
        # =============================================================================
        oos_r2["Step"].append(i)
        #update
        oos_r2["Accuracy"].append(oos_pred(berm_swaptions = sim_oos,
                                           model = out,
                                           step = i, laguerre = True)) #append r2 and update
    
    
    
    results_table = pd.DataFrame({"Step": np.flip(steps),
                                 "Training R2": best_results_r2["Accuracy"],
                                 "OOS R2": oos_r2["Accuracy"],
                                  "Exc Pr": np.sum(sim.index[:, np.flip(steps)], axis = 0)/training_paths,
                                  "Exc Pr OOS": np.sum(sim_oos.index[:, np.flip(steps)], axis = 0)/testing_paths},
                                 )
    
   
    print(results_table)
    
    price = np.sum(sim.cmmf[:, :-1] * np.multiply(sim.index[:,1:], sim.value[:,1:]))/training_paths 
    print(price)
    
    price_oos = np.sum(sim_oos.cmmf[:, :-1] * np.multiply(sim_oos.index[:,1:], sim_oos.value[:,1:]))/testing_paths
    print(price_oos)
    
    
    return (price, price_oos, results_table, total_time, model_list)
  

#%%


#%%


xgb_results_1_10 = []


# with open(path_results+"xgb_results_1_10.pkl", 'rb') as handle:
#         xgb_results_1_10 = pickle.load(handle)




#1 NC 10
xgb_results_1_10.append(berm_xgb(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_10k, strike = 0.003, lockout = 1, tenor = 10, search_iter = 8, cv = 5))

save_object(xgb_results_1_10, path_results+"xgb_results_1_10.pkl")


xgb_results_1_10.append(berm_xgb(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_50k, strike = 0.003, lockout = 1, tenor = 10, search_iter = 8, cv = 5))

save_object(xgb_results_1_10, "xgb_results_1_10.pkl")

xgb_results_1_10.append(berm_xgb(berm_swaptions = berm_swaptions,
                          sim_rates = sim_rates_100k, strike = 0.003, lockout = 1, tenor = 10, search_iter = 8, cv = 5))

save_object(xgb_results_1_10, "xgb_results_1_10.pkl")



# =============================================================================
# SD
# =============================================================================

xgb_1_10_sd = []

for i in range(10):
    sample_sim_rates_20k = sim_rates_100k[:,:,
                        np.random.choice(np.arange(100000), 20000, False)]
    xgb_1_10_sd.append(berm_xgb(berm_swaptions = berm_swaptions,
                          sim_rates = sample_sim_rates_20k, strike = 0.003, lockout = 1, tenor = 10, search_iter = 10, cv = 5))

save_object(xgb_1_10_sd, path_results+"xgb_1_10_sd.pkl")





#1 NC 5

# xgb_results_1_5 = []

# xgb_results_1_5.append(berm_xgb(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_50k, strike = 0.0025, lockout = 1, tenor = 5, search_iter = 5, cv = 5))


# xgb_results_1_5.append(berm_xgb(berm_swaptions = berm_swaptions,
#                           sim_rates = sim_rates_100k, strike = 0.0025, lockout = 1, tenor = 5, search_iter = 5, cv = 5))





