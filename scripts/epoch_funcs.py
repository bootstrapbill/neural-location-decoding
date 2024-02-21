# Author William Turner williamfrancisturner@gmail.com
# 

import mne
import numpy as np    
from mne.time_frequency import tfr_array_morlet
from sklearn.model_selection import train_test_split

def epoch_localizers(raw, pID, tmin=-0.2, tmax=1):
    """Epoch the localiser stimuli"""
    events = mne.find_events(raw, stim_channel='Status', output ='onset', 
                             min_duration = 0.002, consecutive=True)
    
    picks = mne.pick_channels(raw.ch_names, include = [], exclude = ['Status'])
    
    offset = 1/120 # correct event timing as triggers were 1 frame late!         
    stim = events[np.isin(events[:,2], np.linspace(1,40,40))]
    stimEpochs = mne.Epochs(raw, stim, picks=picks,
                             baseline=(-0.2-offset,0-offset), 
                             tmin=tmin-offset, 
                             tmax=tmax-offset,
                             preload=True)
    # mne recommends resampling epochs: https://mne.tools/0.11/auto_examples/preprocessing/plot_resample.html
    stimEpochs.resample(sfreq = 512) # lowpass and resample 
    
    xTrain = stimEpochs.get_data()
    yTrain = stimEpochs.events[:, 2].copy() 
        
    return xTrain, yTrain

def get_just_eoochs(raw, pID, tmin=-0.2, tmax=1):
    """Epoch the localiser stimuli"""
    events = mne.find_events(raw, stim_channel='Status', output ='onset', 
                             min_duration = 0.002, consecutive=True)
    
    picks = mne.pick_channels(raw.ch_names, include = [], exclude = ['Status'])
    
    offset = 1/120 # correct event timing as triggers were 1 frame late!         
    stim = events[np.isin(events[:,2], np.linspace(1,40,40))]
    stimEpochs = mne.Epochs(raw, stim, picks=picks,
                             baseline=(-0.2-offset,0-offset), 
                             tmin=tmin-offset, 
                             tmax=tmax-offset,
                             preload=True)
    # mne recommends resampling epochs: https://mne.tools/0.11/auto_examples/preprocessing/plot_resample.html
    stimEpochs.resample(sfreq = 512) # lowpass and resample 

    return stimEpochs

def epoch_tf(data, pID):
    
    freqs = np.linspace(2,40,20) 
    n_cycles = np.logspace(np.log10(3), 
                           np.log10(10),
                           len(freqs));
    power = tfr_array_morlet(data, sfreq = 512, freqs=freqs, n_cycles=n_cycles,
                             use_fft=True, output='power')
    power = np.ndarray.copy(power[:,:,:,512:1127]) # chop down to [-200, 1000] ms 
    
    return power

def epoch_loc_and_motion(raw, pID, tmin=-0.2, tmax=0.8, synth=False):
 
    events = mne.find_events(raw,stim_channel='Status', output ='onset',
                             min_duration = 0.002, consecutive=True)
    
    picks = mne.pick_channels(raw.ch_names, ['P7', 'P5', 'P3', 'P1', 'Pz', 
                                            'P2', 'P4', 'P6', 'P8', 'PO7',
                                            'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2'])
    
    offset = 1/120 # used to correct event timing as triggers were 1 frame late
    stim = events[np.isin(events[:,2], np.linspace(1,40,40))]     
    stimEpochs = mne.Epochs(raw, stim, picks=picks,
                            baseline=(-0.2-offset,0-offset), 
                            tmin=tmin-offset, 
                            tmax=tmax-offset, 
                            preload=True)  
    stimEpochs.resample(sfreq = 512) 
    
    xTrain = stimEpochs.get_data() 
    yTrain = stimEpochs.events[:, 2].copy() 
    
    if synth:
        return xTrain, yTrain 
    
    xTest = []
    yTest = []
    
    triggers = events[:,2]
    moving = np.array(np.where(np.isin(triggers, range(81,121))))
    
    # mark stops that didn't have a reversal
    for triggerCode in [162, 170]: # loop through stops 
        
        start_stop_rev = np.where(np.isin(triggers, [161, 162, 164, 169, 170, 172]))[0] # start, stop and reversal triggers
        start_stop_rev_trig = triggers[start_stop_rev]
        stops = np.where(np.isin(start_stop_rev_trig, triggerCode))[0]
        pure_stops = np.where(np.isin(start_stop_rev_trig[stops-1], triggerCode - 1))[0]
        events[start_stop_rev[stops[pure_stops]], 2] += 100 # make pure stops distinct
    
    triggerCodes = [[161, 169], [262, 270], [164, 172]] # (start: 161 and 169) (pure stop: 262 and 270) (reversal: 164 and 172)
    for motionDir in [0, 1]:
        for epochType in [0, 1, 2]:
            moving = moving[moving < (len(events) - 1)] # prevent crash if final trigger is a moving trigger
            eventInd = events[moving + 1, 2] == triggerCodes[epochType][motionDir]
            events[moving[eventInd], 2] += 500   
            event = events[np.isin(events[:,2], np.array(range(581,621)))]
            eventEpochs = mne.Epochs(raw, event, picks = picks,
                             baseline = (-1-offset, 0-offset), 
                             tmin= -1-offset,tmax = 1-offset, 
                             preload=True)
            
            eventEpochs.resample(512)
            events[moving[eventInd], 2] -= 500   
    
            xTest.append(eventEpochs.get_data())
            yTestTemp = eventEpochs.events[:, 2].copy() 
            yTest.append((yTestTemp-580)-1) # note, yTest deliberately -1 relative to yTrain (yTest is used as index, not for scoring)
            
    return xTrain, yTrain, xTest, yTest

def create_synthetic(xTrain, yTrain):
    
    xA, xB, yA, yB = train_test_split(xTrain,yTrain,test_size=0.5,shuffle=False)

    positions = np.hstack((np.linspace(1,40,40), np.linspace(1,40,40)))
        
    # Stimulus reaches a new position every 512/40 = 12.8 samples. 
    indices = np.round(np.linspace(12.8, 512, 40)).astype(int) - 1 # subtract one because of zero indexing
    
    trains = [xA, xB, yA, yB]
    tests = [xB, xA, yB, yA]
    
    xTest = []
    xTrain = []
    yTrain = []
    yTest = []
    
    rng = np.random.default_rng(seed = 1)
    
    for rep in [0, 1]:
        
        tempTrain = trains[rep].copy() 
        tempYTrain = trains[rep + 2].copy()
        tempTest = tests[rep].copy()
        tempYTest = tests[rep + 2].copy()
        
        for ind in np.linspace(1,40,40).astype('int'):
            temp = tempTest[tempYTest == ind, :, :]
        
            for i in range(temp.shape[0]):
                superTrial = np.nan * np.zeros([40, 17, tempTrain.shape[2]])
                superTrial[0, :, :] = temp[i, :, :]
            
                for x, subPos in enumerate(positions[ind:(ind + 39)]):
                    randInd = rng.choice(np.argwhere(tempYTest == subPos).flatten())
                    addedSignal = np.hstack((np.nan * np.empty((17, indices[x])), tempTest[randInd, :, :-indices[x]]))
                    superTrial[x + 1, :, :] = addedSignal
            
                temp[i, :, :]  = np.nansum(superTrial, axis = 0) 
        
            tempTest[tempYTest == ind, :, :] = temp 
            
        xTrain.append(tempTrain)
        yTrain.append(tempYTrain) 
        xTest.append(tempTest)
        yTest.append(tempYTest) 
    
    return xTrain, xTest, yTrain, yTest

