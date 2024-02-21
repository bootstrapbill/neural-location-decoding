# Author William Turner williamfrancisturner@gmail.com
# 

import os 
os.chdir('/Users/willturner/Documents/Uni/Postdoc 2021/Decoding_Smooth_Reversals_EEG_data/')

import numpy as np
from matplotlib import pyplot as plt

os.chdir('/Users/willturner/Documents/Uni/Postdoc 2021/Decoding_Smooth_Reversals_EEG_data')
   
from plot_funcs import (loadLDA, loadSynth, weight_mean_circ, smooth_angles)

os.chdir('/Users/willturner/Documents/Uni/Postdoc 2021/Decoding_Smooth_Reversals_EEG_data/data')

pIDs = ["01","02","03","04","05","06","07","08","09",
       "10","11","12","13","14","15","16","17","18"]

allData = loadLDA(pIDs, 'LDA', time_avg = False)
allData[[3,4,5], :, :, :, :] = np.flip(allData[[3,4,5], :, :, :, :], axis = 3)

start = np.mean(np.mean(allData[[0,3],:,:,:,:],axis = 0),axis = 3)
stop = np.mean(np.mean(allData[[1,4],:,:,:,:],axis = 0),axis = 3)
reversal = np.mean(np.mean(allData[[2,5],:,:,:,:],axis = 0),axis = 3)

allDataSynth = loadSynth(pIDs, 'synth')
synthetic = np.mean(allDataSynth, axis = 3)

###### FIGURE 3 #######

scaler = 0.0003
color = "green"
colorMax = "purple"

fig = plt.figure(figsize=(15, 9),dpi=300)
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,5)
ax6 = plt.subplot(2,3,6)

axes = [ax1,ax2,ax3,ax4,ax5,ax6]

allEpochs = [stop,reversal,synthetic]

titles = ['Pre-Stop Data', 'Pre-Reversal Data', 'Autocorrelated Control']
for epoch in [0,1,2]:
    
    # generate marker of stimulus position over time
    pos = np.linspace(1,41,512) 
    pos[pos>40.5] -= 40
    pos = np.hstack((pos[256:], pos[:256])) - 1 # subtract 1 to use as index
    ytimes = np.linspace(0, 500, 257) # training times

    # array for re-aligned map
    alignedMap = np.zeros([257, 512, 40])
    
    tempData = allEpochs[epoch]
    
    for i in range(512):
        position = int(np.round(pos[int(i)]))
        temp = np.concatenate((tempData[:, i, position:], tempData[:, i, :position]),axis = 1)
        alignedMap[:, i, :] = temp 
    
    # recentre data
    alignedMap = np.concatenate((alignedMap[:, :, 20:], alignedMap[:, :, :20]), axis = 2).T
    plotData = np.mean(alignedMap,axis = 1).T
    
    # calculate centroid (and magnitude of vector)
    wa, mag = weight_mean_circ(plotData)
    size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 20 # create size variable from vector length
   
    ignoreStart = 38 # don't smooth data from pre-stim info training timepoints

    maxInd = np.argmax(plotData[ignoreStart:], axis = 1) + 1
    
    smoothed_max = smooth_angles(maxInd)
    smoothed_wa = smooth_angles(wa[ignoreStart:])
    
    # cut out points where position values 'wrap around' (this is purely aesthetic to avoid large horizontal lines)
    smoothed_max[np.hstack((0, np.abs(np.diff(smoothed_max)) > 10)) == 1] = np.nan
    smoothed_wa[np.hstack((0, np.abs(np.diff(smoothed_wa)) > 10)) == 1] = np.nan
    
    if epoch == 2: 
        scaler *= 2.5 # change scaling for synthetic data (which is different SNR)
        
    for plotType in [0,3]:
        
        if plotType == 0: 
            axes[plotType + epoch].matshow(plotData,
                vmin=0.025 - scaler, vmax=0.025 + scaler, cmap='coolwarm', origin='lower',
                extent=[1, 40, 0, 500], aspect='auto')
            axes[plotType + epoch].set_title(titles[epoch], fontsize = 15, fontweight = 'bold')

        else: 
            axes[plotType + epoch].plot(smoothed_max, ytimes[ignoreStart:], color = colorMax)
            axes[plotType + epoch].plot(smoothed_wa, ytimes[ignoreStart:], color = color)

            if epoch == 1: 
                axes[plotType + epoch].set_xlabel('Position', fontsize = 20, fontweight = 'bold')

    
        if epoch == 0:
            axes[plotType + epoch].set_yticks([100, 200, 300, 400])
            axes[plotType + epoch].set_ylabel('Training Time (ms)', fontsize = 20, fontweight = 'bold')
        else: 
            axes[plotType + epoch].set_yticks([])
        
        axes[plotType + epoch].plot([21, 21],[0, 800],'k')
        axes[plotType + epoch].plot([21, 1],[0, 500],'k')
        axes[plotType + epoch].set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
        axes[plotType + epoch].set_ylim([0, 400])
        axes[plotType + epoch].set_xlim([1, 40])
        axes[plotType + epoch].xaxis.set_ticks_position('bottom')

plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=3.5)

plt.savefig('Fig5.png')

###### FIGURE 3 #######

fig = plt.figure(figsize=(10, 9),dpi=300)
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)

axes = [ax1,ax2]

avgData = np.mean(np.stack((stop, reversal), axis = 3), axis = 3)
allEpochs = [avgData, synthetic]

titles = ['Empirical \n Data', 'Autocorrelated \n control']

for epoch in [0,1]:
    
    alignedMap = np.zeros([257, 512, 40])
    
    tempData = allEpochs[epoch]
    
    for i in range(512):
        position = int(np.round(pos[int(i)]))
        temp = np.concatenate((tempData[:, i, position:], tempData[:, i, :position]),axis = 1)
        alignedMap[:, i, :] = temp 
    
    alignedMap = np.concatenate((alignedMap[:, :, 20:], alignedMap[:, :, :20]), axis = 2).T
    plotData = np.mean(alignedMap,axis = 1).T
    
    wa, mag = weight_mean_circ(plotData)    
    size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 20
   
    timeEnd = 205 # to avoid complexity of fitting a line through circular space, 
                  # we simply fit until just before the data wraps
                  
    maxInd = np.argmax(plotData[ignoreStart:], axis = 1) + 1
              
    smoothed_max = smooth_angles(maxInd)
    smoothed_wa = smooth_angles(wa[ignoreStart:])
   
    # take moving average of angles
    smoothed_max[np.hstack((0, np.abs(np.diff(smoothed_max)) > 5)) == 1] = np.nan
    smoothed_wa[np.hstack((0, np.abs(np.diff(smoothed_wa)) > 5)) == 1] = np.nan
    
    # fit line 
    temp = smoothed_max
    temp[temp > 20] = temp[temp > 20] - 40
    y = temp[:timeEnd - ignoreStart]
    x = ytimes[ignoreStart:timeEnd]
    idx = np.isfinite(y)
    a, b = np.polyfit(x[idx], y[idx], 1)
         
    axes[epoch].plot(a*ytimes[ignoreStart:timeEnd] + b, ytimes[ignoreStart:timeEnd], color = colorMax, linewidth = 3)
    axes[epoch].scatter(smoothed_max, ytimes[ignoreStart:], color = colorMax, s = 2)
    # axes[epoch].scatter(smoothed_wa, ytimes[ignoreStart:], color = color, s = 1)

    axes[epoch].set_xlabel('Position', fontsize = 20, fontweight = 'bold')

    if epoch == 0:
        axes[epoch].set_yticks([100, 200, 300, 400, 500])
        axes[epoch].set_ylabel('Training Time (ms)', fontsize = 20, fontweight = 'bold')
    else: 
        axes[epoch].set_yticks([])
        
    axes[epoch].plot([21, 21],[0, 800],'k')
    axes[epoch].plot([21, 1],[0, 500],'k')
    axes[epoch].set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
    axes[epoch].set_ylim([0, 400])
    axes[epoch].set_xlim([1, 40])
    axes[epoch].xaxis.set_ticks_position('bottom')
    axes[epoch].set_title(titles[epoch], fontsize = 20, fontweight = 'bold')
    
plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=3.5)


###### FIGURE 3 #######

scaler = 0.0003

fig = plt.figure(figsize=(15, 8),dpi=300)
ax1 = plt.subplot(2,6,1)
ax2 = plt.subplot(2,6,2)
ax3 = plt.subplot(2,6,3)
ax4 = plt.subplot(2,6,4)
ax5 = plt.subplot(2,6,5)
ax6 = plt.subplot(2,6,6)
ax7 = plt.subplot(2,6,7)
ax8 = plt.subplot(2,6,8)
ax9 = plt.subplot(2,6,9)
ax10 = plt.subplot(2,6,10)
ax11 = plt.subplot(2,6,11)
ax12 = plt.subplot(2,6,12)

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

allEpochs = [start, synthetic]

titles = ['Empirical \n Data', 'Autocorrelated \n control']

for epoch in [0, 1]:
    
    if epoch == 0:
        offset = 511
    else: 
        offset = 0
        
    alignedMap = np.zeros([257, 512, 40])
    
    tempData = allEpochs[epoch]
    
    for i in range(512):
        position = int(np.round(pos[int(i)]))
        temp = np.concatenate((tempData[:, i + offset, position:], tempData[:, i + offset, :position]),axis = 1)
        alignedMap[:, i, :] = temp 
    
    alignedMap = np.concatenate((alignedMap[:, :, 20:], alignedMap[:, :, :20]), axis = 2).T
    #plotData = np.mean(alignedMap,axis = 1).T
    
    #wa, mag = weight_mean_circ(plotData)    
    #size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 20
   
    #timeEnd = 205 # to avoid complexity of fitting a line through circular space, 
                  # we simply fit until just before the data wraps
                  
    #maxInd = np.argmax(plotData[ignoreStart:], axis = 1) + 1
              
    #smoothed_max = smooth_angles(maxInd)
    #smoothed_wa = smooth_angles(wa[ignoreStart:])
   
    # take moving average of angles
    #smoothed_max[np.hstack((0, np.abs(np.diff(smoothed_max)) > 5)) == 1] = np.nan
    #smoothed_wa[np.hstack((0, np.abs(np.diff(smoothed_wa)) > 5)) == 1] = np.nan
    
    # # fit line 
    # temp = smoothed_max
    # temp[temp > 20] = temp[temp > 20] - 40
    # y = temp[:timeEnd - ignoreStart]
    # x = ytimes[ignoreStart:timeEnd]
    # idx = np.isfinite(y)
    # a, b = np.polyfit(x[idx], y[idx], 1)
    times = [80, 100, 140, 220, 300, 500]
     
    if epoch == 1:
        scaler *= 3
        
    for i in range(6):
        
        ind = np.argmin(np.abs(ytimes - times[i]))
        axes[(epoch*6) + i].matshow(np.mean(alignedMap[:, ind - 5:ind + 5, :], axis = 1).T,vmin=0.025 - scaler, vmax=0.025 + scaler, cmap='coolwarm', origin='lower',
                                    extent=[1, 40, 0, 500], aspect='auto')
        
        if i == 0:
            axes[(epoch*6) + i].set_yticks([100, 200, 300, 400, 500])
            axes[(epoch*6) + i].set_ylabel('Training Time (ms)', fontsize = 20, fontweight = 'bold')
        else: 
            axes[(epoch*6) + i].set_yticks([])
        
        if epoch == 1: 
            axes[(epoch*6) + i].set_xlabel('Position', fontsize = 20, fontweight = 'bold')

        axes[(epoch*6) + i].plot([21, 21],[0, 800],'k')
        axes[(epoch*6) + i].plot([21, 1],[0, 500],'k')
        axes[(epoch*6) + i].set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
        axes[(epoch*6) + i].set_ylim([0, 400])
        axes[(epoch*6) + i].set_xlim([1, 40])
        axes[(epoch*6) + i].xaxis.set_ticks_position('bottom')
        if epoch == 0: 
            axes[(epoch*6) + i].set_title(str(times[i]) + ' ms', fontsize = 20, fontweight = 'bold')
    
plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=3.5)

