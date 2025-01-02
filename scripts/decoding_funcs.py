# Author William Turner williamfrancisturner@gmail.com
# 

import mne         
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import SlidingEstimator, GeneralizingEstimator
import pandas as pd

def custom_CV_selector(shape):
    """generate selector indices for custom 5 cross-validation routine"""
    
    repeats = int(shape/5)
    selectorMain = np.repeat([0,1,2,3,4],repeats).astype(int)
    
    for ind in np.argwhere(np.diff(selectorMain) == 1).ravel():
        
        selectorMain[ind + 1:ind + 6] = -2 # remove 5 initial flashes to prevent
                                           # lingering leakage between train and test sets 
    return selectorMain 

def invAngError(y_pred, y): 
    """ Calculate the inverse mean aboslute angular error between predicted and actual pos.
    Circ difference code is adapted from the CircStat MATLAB toolbox, see Berens, 2009.
    This is used to score the circular SVR. Have checked and this is equivalent to 
    the scoring approach used by JR King.
    """
    
    # bring SVR predictions from default arctan2 range back into y range (0, 2pi)    
    ind = y_pred<0
    y_pred[ind] += (np.pi * 2)
    
    diff = np.angle(np.exp(1j*y_pred)/np.exp(1j*y)) # calculate angular difference
    mdiff = np.mean(np.abs(diff)) # take avg absolute difference 
    accuracy = (np.pi/2) - mdiff # centre on zero, range is [+pi to -pi]
    
    return accuracy/np.pi # WARNING: this scores ranges between -0.5 and 0.5,
                          # to rescale between -1 and 1, multiply by 2. 
                          # see plot_SVR.py.
    
def errorPos(y_pred, y): 
    """ Calculate inverse angular error at each possible position.
    Used to evaluate model performance across all stimulus positions.
    This allows us to see if there are spatial biases in the decoding 
    performance. 
    """
    
    # bring SVR predictions from default arctan2 range back into y range (0, 2pi)       
    ind = y_pred<0
    y_pred[ind] += (np.pi * 2)
    
    diff = np.angle(np.exp(1j*y_pred)/np.exp(1j*y)) # calculate angular difference
    df = pd.DataFrame({'error': (np.pi/2)-np.abs(diff),'position': y}) # centre on zero
    accuracy_by_pos = df.groupby('position').mean().to_numpy() # take average

    return accuracy_by_pos/np.pi # WARNING: see note above! range is [-0.5, 0.5]

class CircRegression(BaseEstimator): 
    """SVR function with circular dependent variable. 
    We simplify the problem and predict the sin and cosine of a given 
    stimulus position (angle). 
    Adapted from: 
    https://github.com/kingjr/jr-tools/blob/8a4c9c42a9e36e224279566945e798869904c4c8/jr/gat/classifiers.py#L208
    """
    
    def __init__(self, clf=None):
        import copy        
        if clf is None:
            clf = LinearSVR(loss='squared_epsilon_insensitive', dual=False)
        self.clf = clf
        self.clf_cos = copy.deepcopy(clf)
        self.clf_sin = copy.deepcopy(clf)

    def fit(self, X, y, sample_weight=None):
        sample_weight = compute_sample_weight(class_weight='balanced', y=y)
        sample_weight = (dict() if sample_weight is None
                         else dict(sample_weight=sample_weight))

        if y.ndim == 1:
            y = np.vstack((y, np.ones_like(y))).T
        cos = np.cos(y[:, 0]) * y[:, 1]
        sin = np.sin(y[:, 0]) * y[:, 1]
        self.clf_cos.fit(X, cos, **sample_weight)
        self.clf_sin.fit(X, sin, **sample_weight)
            
    def predict(self, X):
        predict_cos = self.clf_cos.predict(X)
        predict_sin = self.clf_sin.predict(X) 
        predict_angle = np.arctan2(predict_sin, predict_cos)
        return predict_angle # NOTE, this is in the default arctan2 range!
                             # (see: https://www.askpython.com/python-modules/numpy/numpy-arctan2)
    
    def score(self, X, y):
        y_pred = self.predict(X)  
        return invAngError(y_pred, y)
    
    @property
    def coef_(self,):
        return np.array([self.clf_cos.coef_, self.clf_sin.coef_])

def decode_diag(x, y, nPos=40):
    """Run sliding window decoding (equivalent to TGM diagonal).
    x = EEG data. 
    y = index of circular stim positions.
    nPos = number of positions.
    """
    
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[y-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = SlidingEstimator(clf, n_jobs=1)
    
    selector = custom_CV_selector(x.shape[0])
    
    scoresAll = np.zeros((5, x.shape[2])) # splits x time points
    
    for split in range(5):
        
        xTrain = x[selector != split, :, :]
        yTrain = angle[selector != split]

        xTest = x[selector == split, :, :]
        yTest = angle[selector == split]
        
        fittedModel = estimator.fit(xTrain, yTrain)
        scoresAll[split, :] = fittedModel.score(xTest, yTest)
        
    scores = np.mean(scoresAll, axis = 0)
    
    return scores

def error_by_pos(x, y, nPos=40):
    """Run diagonal decoding and calculate error for each stim position.
    xTrain = EEG data. 
    yTrain = index of circular stim positions.
    nPos = number of localizer positions
    """
    
    x = np.copy(x[:, :, 138:231]) # ~70-250 ms 

    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[y-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = SlidingEstimator(clf, n_jobs=1)
    
    selector = custom_CV_selector(x.shape[0])

    scoresAll = np.zeros((5, 40, x.shape[2])) # splits x positions x time points
    
    for split in range(5):
        
        xTrain = x[selector != split, :, :]
        yTrain = angle[selector != split]

        xTest = x[selector == split, :, :]
        yTest = angle[selector == split]
        
        fittedModel = estimator.fit(xTrain, yTrain)
        y_pred = fittedModel.predict(xTest)
        
        scores = np.zeros((40, y_pred.shape[1])) # positions x times
        for time in range(y_pred.shape[1]):
            temp_pred = y_pred[:,time]
            scoresAll[split,:,time] = errorPos(temp_pred, yTest).flatten()
            
    scores = np.mean(scoresAll, axis = 0)  
    
    return np.mean(scores, axis = 1) # average over time

def decode_TGM(x, y, nPos=40):
    """Run temporally-generalized decoding (King & Dehaene, 2014)
    xTrain = EEG data. 
    yTrain = index of circular stim positions.
    nPos = number of localizer positions
    """
    
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[y-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = GeneralizingEstimator(clf, n_jobs=1)
    
    selector = custom_CV_selector(x.shape[0])

    scoresAll = np.zeros((5, x.shape[2], x.shape[2])) # splits x time x time
    
    for split in range(5): 
        
        xTrain = x[selector != split, :, :]
        yTrain = angle[selector != split]

        xTest = x[selector == split, :, :]
        yTest = angle[selector == split]
        
        fittedModel = estimator.fit(xTrain, yTrain)
        scoresAll[split, :, :] = fittedModel.score(xTest, yTest) 
        
    scores = np.mean(scoresAll, axis = 0)
    
    return scores

def searchlight(stimEpochs):
    """Run a searchlight version of the diagonal decoding.
    Loop through each electrode and, including only its immediate neighbours, 
    re-run the diag decoding analysis and return peak decoding accuracy.
    """
       
    scoresAll = np.zeros((64, 93)) # chans x time points 

    adj, names = mne.channels.find_ch_adjacency(stimEpochs.info,ch_type='eeg')

    for k in range(len(names)): 
        
        neigh_idx = np.where(adj[k,:].toarray().ravel())[0]
        neigh_names = [names[i] for i in neigh_idx]
        
        xTrain = stimEpochs.get_data(picks=neigh_names)
        xTrain = np.ndarray.copy(xTrain[:, :, 138:231]) # ~70-250 
        yTrain = stimEpochs.events[:, 2].copy()
        
        scoresAll[k, :] = decode_diag(xTrain, yTrain)
        
    return scoresAll

def decode_diag_tf(x, y):
    """Run diagonal decoding on time-frequency data.
    """
    
    scoresAll = np.zeros((20, 615)) # freqs x times
    
    for freq in range(x.shape[2]):
        
        xTrain = x[:,:,freq,:]
        scoresAll[freq, :] = decode_diag(xTrain,y)
        
    return scoresAll

def decode_successive(x, y, nPos = 40):
    """Run TG decoding, predicting the position of successive stimuli.
    We repeat this across splits of train and test data. 
    """
    
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[y-1])    
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    trainedClassifier = GeneralizingEstimator(clf, n_jobs=1)
    
    selectorMain = custom_CV_selector(x.shape[0])

    scoresAll = np.zeros((5, 5, x.shape[2]-102, x.shape[2])) # splits, stim (0 to N + 4), train times, test times 
    
    for split in range(5):
        
        xTrain = np.ndarray.copy(x[selectorMain != split,:,102:]) # chop down to 0-500 ms 
        yTrain = angle[selectorMain != split]
        
        xTest = x[selectorMain == split,:,:] 
        yTest = angle[selectorMain == split]
        
        trainedClassifier.fit(X = xTrain, y = yTrain)
                
        repeats = int(xTest.shape[0]/5)
        selector = np.tile([0,1,2,3,4],repeats).astype(int)
        selector = np.hstack((selector, np.nan * np.ones(len(yTest) - len(selector)))) # nan pad if needed
        
        for stim in [0,1,2,3,4]:
            scoresAll[split, stim, :, :] = trainedClassifier.score(X=xTest[selector == stim, :, :], 
                                                                       y=yTest[selector == stim])
    return scoresAll
    
def centre_map(scores, y):
    """Re-centre probability values
    """
    
    for ind in range(0, len(y)):
        
        scores[ind,:,:,:] = np.concatenate((scores[ind,:,:,y[ind]:], scores[ind,:,:,:y[ind]]),axis=2)

    scores = np.mean(scores,axis=0)
    scores = np.concatenate((scores[:,:,20:], scores[:,:,:20]), axis = 2)
    
    return scores

def LDA_map(xTrain, yTrain, xTest, yTest, synth=False, nPos=40):
    """Extract a probabilistic map over stimulus positions via LDA.
    We pre-train the models on the localizer data and then use the 
    predict_proba function to get a probability for each class (position).
    Note, we seperately scale the testing and training data... this is slightly 
    unconventional, but means we are predicting from the relative pattern of 
    voltage across electrodes (for a given timepoint). This is purely to keep 
    things consistent when we analyse the synthetic data, where its necessary to use 
    seperate scaling. 
    """
    
    xTrain = np.ndarray.copy(xTrain[:, :, 102:]) #  0-400 ms
       
    scaler = StandardScaler()
    
    for i in range(xTrain.shape[2]):
        
        xTrain[:,:,i] = scaler.fit_transform(xTrain[:,:,i]) 
                                                                  
    clf = make_pipeline(PCA(n_components = 0.99), 
                        LinearDiscriminantAnalysis(priors = np.tile(1/nPos, nPos)))    
    
    trainedClassifier = GeneralizingEstimator(clf, n_jobs=1)
    trainedClassifier.fit(xTrain, yTrain)
    
    decodingScores = np.zeros((len(xTest), xTrain.shape[2], xTest[0].shape[2], nPos))
    
    for trialType in range(len(xTest)):
        
        for i in range(xTest[trialType].shape[2]):
            
            xTest[trialType][:,:,i] = scaler.fit_transform(xTest[trialType][:,:,i])
            
        scores = trainedClassifier.predict_proba(xTest[trialType])
        decodingScores[trialType, :, :, :] = centre_map(scores, yTest[trialType])
        
    return decodingScores

def LDA_map_synth(xTrain, yTrain, xTest, yTest, synth=False, nPos=40):
    """Same as LDA_map but for synthetic data. 
    """
    
    decodingScores = np.zeros((len(xTest), 257, 564, nPos))

    for rep in [0, 1]:
        
        train = np.ndarray.copy(xTrain[rep][:, :, 102:359]) # cut down to 0-500 ms
        test = xTest[rep][:, :, 102:666] # only need to decode from 0-1100 ms
        
        scaler = StandardScaler()
        
        for i in range(train.shape[2]):
            
            train[:,:,i] = scaler.fit_transform(train[:,:,i]) 
                                                                  
        clf = make_pipeline(PCA(n_components = 0.99), 
                        LinearDiscriminantAnalysis(priors = np.tile(1/nPos, nPos)))    
    
        trainedClassifier = GeneralizingEstimator(clf, n_jobs=2)
        trainedClassifier.fit(train, yTrain[rep])
    
        for i in range(test.shape[2]):
            
            test[:,:,i] = scaler.fit_transform(test[:,:,i])
        
        scores = trainedClassifier.predict_proba(test)

        decodingScores[rep, :, :, :] = centre_map(scores, yTest[rep])
        
    return decodingScores
