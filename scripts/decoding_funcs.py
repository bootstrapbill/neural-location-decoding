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
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from sklearn.model_selection import cross_val_predict
import pandas as pd

def invAngError(y_pred, y): 
    """ Calculate the inverse mean aboslute angular error between predicted and actual pos.
    Circ difference code is adapted from the CircStat MATLAB toolbox, see Berens, 2009.
    This is used to score the circular SVR. Have checked and this is equivalent to 
    the scoring approach used by JR King.
    """
    # SVR predictions are in the default arctan2 range, so shift back to [0, 2pi]   
    ind = y_pred<0
    y_pred[ind] += (np.pi * 2)
    
    diff = np.angle(np.exp(1j*y_pred)/np.exp(1j*y)) # calculate angular difference
    mdiff = np.mean(np.abs(diff)) # take avg absolute difference 
    accuracy = (np.pi/2) - mdiff # Shift so 0 is chance. range is [+pi to -pi]
    
    return accuracy
    
def errorPos(y_pred, y): 
    """ Calculate inverse angular error at each possible position.
    Used to evaluate model performance across all stimulus positions.
    This allows us to see if there are spatial biases in the decoding 
    performance. 
    """
    # bring SVR predictions, which are in the default arctan2 range, to normal 
    # radian range (0, 2pi)    
    ind = y_pred<0
    y_pred[ind] += (np.pi * 2)
    
    diff = np.angle(np.exp(1j*y_pred)/np.exp(1j*y)) 
    df = pd.DataFrame({'error': (np.pi/2)-np.abs(diff),'position': y})
    accuracy_by_pos = df.groupby('position').mean().to_numpy()

    return accuracy_by_pos

class CircRegression(BaseEstimator): 
    """SVR function with circular dependent variable. 
    We simplify the problem and predict the sin and cosine of a given 
    stimulus position (angle). 
    This is adapted from: 
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

def decode_diag(xTrain, yTrain, nPos=40):
    """Run decoding along just the diagonal. 
    xTrain = EEG data. 
    yTrain = index of circular stim positions.
    nPos = number of localizer positions.
    """
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[yTrain-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = SlidingEstimator(clf, n_jobs=1)
    scores = cross_val_multiscore(estimator, X=xTrain, y=angle, cv=5, n_jobs=1)
    scores = np.mean(scores, axis = 0)
    
    return scores

def error_by_pos(xTrain, yTrain, nPos=40):
    """Run diagonal decoding and calculate error for each stim position.
    xTrain = EEG data. 
    yTrain = index of circular stim positions.
    nPos = number of localizer positions
    """
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[yTrain-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = SlidingEstimator(clf, n_jobs=1)
    xTrain = np.ndarray.copy(xTrain[:, :, 128:231]) # predict between ~50-250 ms
                                                    # note, remember the 8 ms offset (if checking these times... 
                                                    # also that last index is not included in slice)

    y_pred = cross_val_predict(estimator, X=xTrain, y=angle, cv=5, n_jobs=1) 
    scores = np.zeros((40, y_pred.shape[1]))
    for time in range(y_pred.shape[1]):
        temp_pred = y_pred[:,time]
        scores[:,time] = errorPos(temp_pred,angle).flatten()
        
    return np.mean(scores, axis = 1)

def decode_diag_tf(xTrain, yTrain):
    """Run diagonal decoding on time-frequency data.
    """
    
    scoresAll = np.zeros((20, 615))
    for freq in range(xTrain.shape[2]):
        x = xTrain[:,:,freq,:]
        scoresAll[freq, :] = decode_diag(x,yTrain)
        
    return scoresAll

def decode_TGM(xTrain, yTrain, nPos=40):
    """Run temporally generalized decoding (King & Dehaene, 2014)
    xTrain = EEG data. 
    yTrain = index of circular stim positions.
    nPos = number of localizer positions
    """
    degrees = np.linspace(360/nPos, 360, nPos)
    angle = np.radians(degrees[yTrain-1])
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    estimator = GeneralizingEstimator(clf, n_jobs=1)
    scores = cross_val_multiscore(estimator, X=xTrain, y=angle, cv=5, n_jobs=1)
    scores = np.mean(scores, axis = 0)
    
    return scores

def searchlight(stimEpochs):
    """Run a searchlight version of the diagonal decoding.
    Loop through each electrode and, including only its immediate neighbours, 
    re-run the diag decoding analysis and return peak decoding accuracy.
    """
       
    scoresAll = np.zeros((64, 103))

    adj, names = mne.channels.find_ch_adjacency(stimEpochs.info,ch_type='eeg')

    for k in range(len(names)): 
        
        neigh_idx = np.where(adj[k,:].toarray().ravel())[0]
        neigh_names = [names[i] for i in neigh_idx]
        
        xTrain = stimEpochs.get_data(picks=neigh_names)
        xTrain = np.ndarray.copy(xTrain[:, :, 128:231]) # chop down to ~50-250, see note about timing above. 
        yTrain = stimEpochs.events[:, 2].copy()
        
        scoresAll[k, :] = decode_diag(xTrain, yTrain)
        
    return scoresAll

def decode_successive(x, y, nPos = 40):
    """Run TG decoding, predicting the position of successive stimuli.
    We repeat this across splits of train and test data. 
    """
    degrees = np.radians(np.linspace(360/nPos, 360, nPos))
    
    clf = make_pipeline(StandardScaler(), PCA(n_components=0.99), CircRegression())
    trainedClassifier = GeneralizingEstimator(clf, n_jobs=1)
    
    repeats = int(x.shape[0]/10)
    selectorMain = np.repeat([0,1,2,3,4,5,6,7,8,9],repeats).astype(int)
    scoresAll = np.zeros((5, 5, x.shape[2]-102, x.shape[2])) # splits, stim (0 to N + 4), train times, test times 
    
    for split in range(5): # essentially how many folds of cross validation
        
        xTrain = np.ndarray.copy(x[selectorMain != split,:,102:]) # chop down to 0-500 ms 
        yTrain = y[selectorMain != split]
        xTest = x[selectorMain == split,:,:] 
        yTest = y[selectorMain == split]
            
        angleTrain = degrees[yTrain-1]
        angleTest = degrees[yTest-1]    
        
        trainedClassifier.fit(X = xTrain, y = angleTrain)
                
        repeats = int(xTest.shape[0]/5)
        selector = np.tile([0,1,2,3,4],repeats).astype(int)
        selector = np.hstack((selector, np.nan * np.ones(len(angleTest) - len(selector)))) # nan pad if needed
        
        for nextStim in [0, 1, 2, 3, 4]:
            scoresAll[split, nextStim, :, :] = trainedClassifier.score(X=xTest[selector == nextStim, :, :], 
                                                                       y=angleTest[selector == nextStim])
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
    voltage across electrodes (for a given timepoint).
    """
    xTrain = np.ndarray.copy(xTrain[:, :, 102:]) # cut down to 0-500 ms
       
    scaler = StandardScaler()
    for i in range(xTrain.shape[2]):
        xTrain[:,:,i] = scaler.fit_transform(xTrain[:,:,i]) 
                                                                  
    clf = make_pipeline(PCA(n_components = 0.99), 
                        LinearDiscriminantAnalysis(priors = np.tile(1/nPos, nPos)))    
    
    trainedClassifier = GeneralizingEstimator(clf, n_jobs=2)
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

