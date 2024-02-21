# Author William Turner williamfrancisturner@gmail.com
# 

import numpy as np
import pickle 
from mne.stats import spatio_temporal_cluster_1samp_test
import scipy

#### SVR PLOTTING FUNCTIONS ##### 

def readData(dataType, pID, session):
    fileName = dataType + '/ID' + str(pID) + '_Session' + str(session) + '_' + dataType + '.pickle'
    with open(fileName, 'rb') as input_file:
        data = pickle.load(input_file) 
    
    return data 

def loadDiag(pIDs):
    allData = np.nan * np.zeros([len(pIDs), 615])
    
    for x, pID in enumerate(pIDs):     
        temp = []  
        
        for session in [1,2]:
            data = readData('diag', pID, session)
            
            if session == 1: 
                temp = data[:615]
            else:
                temp = np.vstack((temp, data[:615]))    

        allData[x, :] = np.mean(temp, axis = 0)
           
    return allData

def loadPos(pIDs):
    allData = np.nan * np.zeros([len(pIDs), 40])
    
    for x, pID in enumerate(pIDs):     
        temp = []  
        
        for session in [1,2]:
            data = readData('pos', pID, session)
            
            if session == 1: 
                temp = data
            else:
                temp = np.vstack((temp, data))    

        allData[x, :] = np.mean(temp, axis = 0)
                
    return allData

def loadTGM(pIDs):
    allData = np.nan * np.zeros([len(pIDs), 615, 615])
    
    for x, pID in enumerate(pIDs): 
        temp = []
       
        for session in [1,2]:
            data = readData('tgm', pID, session)   
                
            if session == 1: 
                temp  = data[:615, :615]
            else:
                temp = np.mean(np.dstack((temp, data[:615, :615])), axis = 2)       

        allData[x, :, :] = temp
                    
    return allData    

def loadSearch(pIDs):
    allData = np.nan * np.zeros([len(pIDs), 64])
    
    for x, pID in enumerate(pIDs): 
        temp = []
       
        for session in [1,2]:
            data = readData('search', pID, session)   
                
            if session == 1: 
                temp = data[:,:103]
            else:
                temp = np.mean(np.dstack((temp, data[:,:103])), axis = 2)

        allData[x, :] = np.mean(temp, axis = 1)
             
    return allData    

def loadTF(pIDs):
    allData = np.nan * np.zeros([len(pIDs), 20, 615])

    for x, pID in enumerate(pIDs):     
        temp = []  
            
        for session in [1,2]:
            data = readData('tf', pID, session)   
        
            if session == 1:
                temp  = data[:, :615]
            else:
                temp = np.dstack((temp, data[:, :615]))       
                
        allData[x, :, :] = np.mean(temp, axis = 2)
        
    return allData

def loadStacked(pIDs):
    allData = np.nan * np.zeros([5, 257, 359, len(pIDs)])

    for x, pID in enumerate(pIDs):     
        temp = []  
            
        for session in [1,2]:
            data = readData('stacked', pID, session)   
        
            if session == 1:
                temp  = np.mean(data[:, :, :257, :359], axis = 0)
            else:
                temp = np.stack((temp, np.mean(data[:, :, :257, :359], axis = 0)), axis = 3)       
                
        allData[:,:,:,x] = np.mean(temp, axis = 3)

    return allData

    
#### LDA PLOTTING FUNCTIONS ####

def loadLDA(pIDs, dataType, time_avg = True):
    if time_avg: 
        ind = [39, 65] 
        allData = np.nan * np.zeros([6, 1024, 40, len(pIDs)])
    else:
        allData = np.nan * np.zeros([6, 257, 1024, 40, len(pIDs)])

    for x, pID in enumerate(pIDs):     
        print(pID)
        
        for session in [1,2]:
            fileName = 'LDA' + '/ID' + str(pID) + '_Session' + str(session) + '_' + dataType + '.pickle'
            with open(fileName, 'rb') as input_file:
                data = pickle.load(input_file) 
                
            data = np.stack(data)
            data = data[:,:257,:1024,:]
            
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

def loadSynth(pIDs, dataType):
    
    allData = np.nan * np.zeros([257, 564, 40, len(pIDs)])

    for x, pID in enumerate(pIDs):     
        print(pID)
        
        for session in [1,2]:
            fileName = 'LDA' + '/ID' + str(pID) + '_Session' + str(session) + '_' + dataType + '.pickle'
            with open(fileName, 'rb') as input_file:
                data = pickle.load(input_file) 
                
            # data = np.mean(data[:, :257,:564,:], axis = 0)
            data = data[1, :257,:564,:]
           
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
    
    return pos, mag

def smooth_angles(pos, window_size=5):
    
    angles = np.linspace(0, 2*np.pi, 40)
    
    smoothed_angles = np.zeros_like(pos, dtype=float)
    angles_to_ind = np.zeros(len(pos))
    
    for x, p in enumerate(pos):
        angles_to_ind[x] = angles[p-1]
        
    for i in range(len(pos)):
        
        # extract a window of angular data
        window = angles_to_ind[i - window_size // 2: i + window_size // 2 + 1]

        # calculate circular mean
        circular_mean = np.angle(np.sum(np.exp(np.multiply(1j, window))))
        
        # convert back to range [0, 2pi]
        if circular_mean < 0:
            circular_mean += 2*np.pi
        
        smooth_pos = np.argmin(np.abs(angles - circular_mean))
        
        # Convert back to degrees
        smoothed_angles[i] = smooth_pos + 1

    return smoothed_angles

#### STATS FUNCTIONS #### 

def cluster_correct(X):
    """Statistical test applied across subjects"""
    # check input
    X = np.array(X)
    X = X[:, :, None] if X.ndim == 2 else X
    
    alpha = 0.01 # cluster-forming threshold
    degrees_of_freedom = 18 - 1 # N = 18
    t_thresh = scipy.stats.t.ppf(1 - alpha/2, df=degrees_of_freedom)

    # stats function report p_value for each cluster
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
        X, threshold = t_thresh, out_type='mask', n_permutations=10000, n_jobs=-1, 
        verbose=True)
    
    # format p_values to get same dimensionality as X
    p_values_ = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, p_values):
        p_values_[cluster.T] = pval
        
    return np.squeeze(p_values_)
