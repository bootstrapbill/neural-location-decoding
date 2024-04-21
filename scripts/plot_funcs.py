# Author William Turner williamfrancisturner@gmail.com
# 

import numpy as np
import pickle 
from mne.stats import spatio_temporal_cluster_1samp_test
import scipy

def read_data(dataType, pID, session):
    
    fileName = dataType + '/ID' + str(pID) + '_Session' + str(session) + '_' + dataType + '.pickle'
    with open(fileName, 'rb') as input_file:
        data = pickle.load(input_file) 
    
    return data 

def load_data(pIDs, dataType):
    
    if dataType == 'diag':
        allData = np.nan * np.zeros([len(pIDs), 615])
    elif dataType == 'pos':
        allData = np.nan * np.zeros([len(pIDs), 40])
    elif dataType == 'tgm':
        allData = np.nan * np.zeros([len(pIDs), 615, 615])
    elif dataType == 'search':  
        allData = np.nan * np.zeros([len(pIDs), 64])
    elif dataType == 'tf':
        allData = np.nan * np.zeros([len(pIDs), 20, 615])
    elif dataType == 'stacked':
        allData = np.nan * np.zeros([len(pIDs), 5, 257, 359])

    for x, pID in enumerate(pIDs):     
            
        temp = []  
        for session in [1,2]:
                
            data = read_data(dataType,pID,session)
            
            if session == 1: 
                
                if dataType == 'diag' or dataType == 'tf':
                    temp = data[:615]
                elif dataType == 'pos':
                    temp = data
                elif dataType == 'tgm':
                    temp  = data[:615, :615]
                elif dataType == 'search':  
                    temp = np.mean(data[:,:103], axis = 1)
                elif dataType == 'tf':
                    temp  = data[:, :615]
                elif dataType == 'stacked':
                    temp  = np.mean(data[:, :, :257, :359], axis = 0)
                
            elif session == 2:

                if dataType == 'diag':
                    temp = np.mean(np.vstack((temp, data[:615])), axis = 0)
                elif dataType == 'pos':
                    temp = np.mean(np.vstack((temp, data)), axis = 0)  
                elif dataType == 'tgm':
                    temp = np.mean(np.dstack((temp, data[:615, :615])), axis = 2)       
                elif dataType == 'search':  
                    temp = np.mean(np.dstack((temp, np.mean(data[:,:103], axis = 1))), axis = 2)
                elif dataType == 'tf':
                    temp  = np.mean(np.dstack((temp, data[:, :615])), axis = 2)       
                elif dataType == 'stacked':
                    temp = np.mean(np.stack((temp, np.mean(data[:, :, :257, :359], axis = 0)), axis = 3), axis = 3)    
                
        allData[x, :] = temp
               
    return allData 
    
def load_LDA(pIDs, time_avg = True):
    
    if time_avg: # if averaging over training times 
        
        ind = [38, 65] # ~75-125 ms
        allData = np.nan * np.zeros([6, 1024, 40, len(pIDs)])
    
    else:
        
        allData = np.nan * np.zeros([6, 206, 1024, 40, len(pIDs)])

    for x, pID in enumerate(pIDs):     
        print(pID)
        
        for session in [1,2]:
            fileName = 'LDA' + '/ID' + str(pID) + '_Session' + str(session) + '_LDA' + '.pickle'
            
            with open(fileName, 'rb') as input_file:
                data = pickle.load(input_file) 
                
            data = np.stack(data)
            data = data[:,:206,:1024,:]
            
            if time_avg:
                if session == 1:   
                    temp = np.mean(data[:,ind[0]:ind[1],:,:], axis = 1)
                else:
                    temp = np.stack((temp,np.mean(data[:,ind[0]:ind[1],:,:], axis = 1)), axis=3)
            else:
                if session == 1:   
                    temp = data
                else:
                    temp = np.stack((temp,data), axis=4)
                    
        if time_avg:    
            allData[:,:,:,x] = np.mean(temp, axis = 3)
        else:
            allData[:,:,:,:,x] = np.mean(temp, axis = 4)

    return allData

def load_synth(pIDs):
    
    allData = np.nan * np.zeros([206, 564, 40, len(pIDs)])

    for x, pID in enumerate(pIDs):     
        print(pID)
        
        for session in [1,2]:
            fileName = 'LDA' + '/ID' + str(pID) + '_Session' + str(session) + '_synth' + '.pickle'
            with open(fileName, 'rb') as input_file:
                data = pickle.load(input_file) 
                
            # data = np.mean(data[:, :206,:564,:], axis = 0)
            data = data[1, :206,:564,:]
           
            if session == 1:   
                temp = data
            else:
                temp = np.stack((temp,data), axis=3)
        
        allData[:,:,:,x] = np.mean(temp, axis = 3)

    return allData

def weight_mean_circ(data):
    """Calculate circular weighted mean (centroid)"""
    
    coords = np.radians(np.linspace(9, 360, 40))
    
    if len(np.shape(data)) > 1:
        coords = np.repeat(coords[np.newaxis, :], data.shape[0], axis = 0)
    
    t = np.multiply(data, np.exp(np.multiply(1j, coords)))
    
    if len(np.shape(data)) > 1:
        r = np.sum(t, axis = 1)
    else: 
        r = np.sum(t, axis = 0)
        
    mu = np.array(np.angle(r))
    
    # convert back to range [0, 2pi]
    mu[mu < 0] += 2*np.pi
        
    mag = np.abs(r)

    pos = np.argmin(abs(np.subtract(mu.T,coords.T)), axis = 0) + 1 
                                                             # + 1 to shift pos 0 to pos 1 etc. 
    
    return pos, mag

def align_data(data, slices):
           
    #  marker of stimulus position over time
    pos = np.linspace(1,41,512) 
    pos[pos>40.5] -= 40
    pos = np.hstack((pos[256:], pos[:256])) - 1 # subtract 1 to use as index
       
    # array for re-aligned map
    alignedMap = np.zeros([206, 512, 40])
          
    for i in slices:
           
        position = int(np.round(pos[int(i)]))
        temp = np.concatenate((data[:, i, position:], data[:, i, :position]),axis = 1)
        alignedMap[:, i, :] = temp 
       
    # re-center data
    alignedMap = np.concatenate((alignedMap[:, :, 20:], alignedMap[:, :, :20]), axis = 2)
       
    return alignedMap

#### STATS FUNCTIONS #### 

def calc_FWHM(data):
    """Estimate FWHM"""

    chance = 1/40 
    
    # calculate estimate of FWHM: 
    half_max = chance + ((data.max() - chance)/2)
    diffs = abs(data-half_max)
    half_max_1 = np.argmin(diffs)
    diffs[half_max_1] = 100
    half_max_2 = np.argmin(diffs)
    width = abs(half_max_1-half_max_2)
    print('FWHM is ' + str(width * 9) + ' polar degrees')

def cluster_correct(X):
    """Statistical test applied across participants"""
    
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    
    alpha = 0.01 # cluster-forming threshold
    degrees_of_freedom = 18 - 1 # N = 18
    t_thresh = scipy.stats.t.ppf(1 - alpha/2, df=degrees_of_freedom)

    # stats function report p_value for each cluster
    # see: https://github.com/kingjr/decod_unseen_maintenance/blob/master/notebook/method_statistics.ipynb
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, threshold = t_thresh, out_type='mask', n_permutations=2**12, n_jobs=-1, 
        verbose=False)
    
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
        
    return np.squeeze(p_values_)
