# Author William Turner williamfrancisturner@gmail.com
# 

import os 
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
import pandas as pd 
from mpl_toolkits.axes_grid1 import make_axes_locatable   
from plot_funcs import load_LDA, load_synth, align_data #, weight_mean_circ # uncomment for centroid plot

os.chdir('/data')

pIDs = ["01","02","03","04","05","06","07","08","09",
       "10","11","12","13","14","15","16","17","18"]
       
allData = load_LDA(pIDs, time_avg = False)
allData[[3,4,5], :, :, :, :] = np.flip(allData[[3,4,5], :, :, :, :], axis = 3)

slices = np.linspace(0, 511, 512).astype('int')

stop = np.mean(np.mean(allData[[1,4],:,:512,:,:],axis = 0), axis = 3) # avg over directions and participants
stop = np.mean(align_data(stop, slices), axis = 1) # re-algined then average over time 

reversal = np.mean(np.mean(allData[[2,5],:,:512,:,:],axis = 0), axis = 3) # avg over directions and participants
reversal = np.mean(align_data(reversal, slices), axis = 1) # re-algined then average over time 

allSynth = np.mean(load_synth(pIDs), axis = 3)
synthetic = np.mean(align_data(allSynth, slices), axis = 1) # re-algined then average over time 


##### Figure 4 ######

scaler = 0.0003
colorMax = "white"
colorLine = "white"

fig = plt.figure(figsize=(13, 5),dpi=300)
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

axes = [ax1,ax2]

# average across stop and reversal
avgData = np.mean(np.stack((stop, reversal), axis = 2), axis = 2)

allEpochs = [avgData,synthetic]

xAll = []
yAll = []

titles = ['Mid-Motion', 'Autocorrelated Control']
for epoch in [0,1]:
    
    plotData = allEpochs[epoch]
    times = np.linspace(0, 400, 206) # training times
    
    ignoreStart = 38 # ~75 ms
    
    posEstimate = np.argmax(plotData[ignoreStart:, :], axis = 1) + 1
    maxVal = np.max(plotData[ignoreStart:, :], axis = 1)

    # uncomment this line to use the centroid rather than the peak estimate (Supplement 1)
    # posEstimate, _ = weight_mean_circ(plotData[ignoreStart:, :])
    
    temp = posEstimate.copy()
    temp[temp > 30] -= 40 # flipping peaks which have wrapped around (x-axis is circular) 
    y = temp
    x = times[ignoreStart:]
    yAll.append(y)
    xAll.append(x)
    
    if epoch == 1: 
        scaler *= 4 # change scaling for synthetic data (which has different SNR)
        
    im = axes[epoch].matshow(plotData,
            vmin=(1/40) - scaler, vmax=(1/40) + scaler, 
            cmap='coolwarm', origin='lower',
            extent=[0.5, 40.5, 0, 400], aspect='auto', 
            interpolation = 'gaussian')
    
    axes[epoch].set_title(titles[epoch], fontsize = 20, fontweight = 'bold')

    if epoch == 0:
        axes[epoch].set_yticks([100, 200, 300, 400])
        axes[epoch].set_ylabel('Training Time (ms)', fontsize = 18, fontweight = 'bold')
    else: 
        axes[epoch].set_yticks([])
        
    axes[epoch].plot([21, 21],[0, 400], 'black', linestyle = '--', linewidth = 3)
    axes[epoch].plot([21, 1],[0, 500], 'black', linestyle = '--', linewidth = 3)
    
    axes[epoch].set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
    axes[epoch].set_ylim([1, 400])
    axes[epoch].set_xlim([0.5, 40.5])
    axes[epoch].xaxis.set_ticks_position('bottom')
    axes[epoch].set_xlabel('Position', fontsize = 18, fontweight = 'bold')

    # rescale max in interval [1, 100]
    size = np.round((100-1)*((maxVal-np.nanmin(maxVal))/(np.nanmax(maxVal) - np.nanmin(maxVal))) + 1)
    axes[epoch].scatter(posEstimate[::8], times[ignoreStart:][::8], s = size[::8], color = colorMax) # sample every ~15 ms
        
    divider = make_axes_locatable(axes[epoch])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([0.025 - scaler, 0.025, 0.025 + scaler])
    cbar.set_ticklabels(['-' + str(round(scaler*10000)), '', 
                         str(round(scaler*10000))])
    
    cbar.ax.set_ylabel('Position Evidence (1e-4)',
                       rotation=270,
                       fontsize=14,
                       fontweight='bold')
    
plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=3.5)

# regression model to test interaction
x1 = np.stack(xAll).ravel()
x2 = np.hstack((np.repeat(0, len(x1)/2), np.repeat(1, len(x1)/2)))
y = np.stack(yAll).ravel()
data = np.vstack((np.vstack((y, x1)), x2)).T

df = pd.DataFrame(data, columns=['Y', 'X1', 'X2'])
model = ols("Y ~ X1 * C(X2)", data=df)
results = model.fit() # results.summary() will print out results table
predictions = results.get_prediction(df)
pred = predictions.summary_frame()
ci_low = pred['mean_ci_lower'] 
ci_upp = pred['mean_ci_upper']
predicted = pred['mean']

# overlay the model fits 
axes[0].plot(predicted[x2==0], x1[x2 == 0], color = colorLine, linewidth = 2)
axes[1].plot(predicted[x2==1], x1[x2 == 1], color = colorLine, linewidth = 2)
axes[0].fill_betweenx(x1[x2 == 0], ci_low[x2==0], ci_upp[x2==0], color = colorLine, alpha = 0.4) 
axes[1].fill_betweenx(x1[x2 == 1], ci_low[x2==1], ci_upp[x2==1], color = colorLine, alpha = 0.4) 

# uncomment to see ratio calculation
#axes[0].scatter(10.75, 400, color = 'red', s = 100) # last of predicted[x2==0]
#axes[0].scatter(21, 400, color = 'green', s = 100) # real - time pos
#axes[0].scatter(5, 400, color = 'blue', s = 100) # 21 - (400/25) to get lagged position
#b = 10.75 - 5
#a = 21 - 5
#ratio = b/a

axes[0].text(3.7, 272, "No compensation", rotation = -66, fontsize = 10)
axes[0].text(21.5, 262, "Full compensation", rotation = -90, fontsize = 10)

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)


## Supplement 2

synthetic = align_data(allSynth, slices)
start = align_data(np.mean(np.mean(allData[[0,3],:,512:,:,:],axis = 0), axis = 3), slices)

scaler = 0.0003
colorMax = "white"

fig = plt.figure(figsize=(13, 7.5),dpi=300)
ax1 = plt.subplot(2,5,1)
ax2 = plt.subplot(2,5,2)
ax3 = plt.subplot(2,5,3)
ax4 = plt.subplot(2,5,4)
ax5 = plt.subplot(2,5,5)
ax6 = plt.subplot(2,5,6)
ax7 = plt.subplot(2,5,7)
ax8 = plt.subplot(2,5,8)
ax9 = plt.subplot(2,5,9)
ax10 = plt.subplot(2,5,10)

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]

allEpochs = [start, synthetic]

times = [80, 100, 150 , 200, 250]

for epoch in [0,1]:
    
    ytimes = np.linspace(0, 400, 206) # training times
    
    if epoch == 1: 
        
        scaler *= 4 # change scaling for synthetic data (which is different SNR)
        
    for x, timeWindow in enumerate([41, 51, 77, 103, 128]):
        
        plotData = np.mean(allEpochs[epoch][:, timeWindow-10:timeWindow+10, :], axis = 1)
    
        ignoreStart = 36 # don't smooth data from pre-stim info training timepoints
        
        maxInd = np.argmax(plotData[ignoreStart:], axis = 1) + 1
        maxVal = np.max(plotData[ignoreStart:], axis = 1) 

        im = axes[x + epoch*5].matshow(plotData,
                        vmin=(1/40) - scaler, vmax=(1/40) + scaler, 
                        cmap='coolwarm', origin='lower',
                        extent=[0.5, 40.5, 0, 400], aspect='auto', 
                        interpolation = 'gaussian')

        size = np.round((100-1)*((maxVal-np.nanmin(maxVal))/(np.nanmax(maxVal) - np.nanmin(maxVal))) + 1)
        size[maxVal < (1/40) + scaler] = np.nan
        
        axes[x + epoch*5].scatter(maxInd[::8], ytimes[ignoreStart:][::8], s = size[::8], color = colorMax)

        if x == 0:
            axes[x + epoch*5].set_yticks([100, 200, 300])
            axes[x + epoch*5].set_ylabel('Training Time (ms)', fontsize = 18, fontweight = 'bold')
        else: 
            axes[x + epoch*5].set_yticks([])
            
        if x == 2 and epoch == 1:
            axes[x + epoch*5].set_xlabel('Position', fontsize = 18, fontweight = 'bold')
       
        if x == 4: 
            divider = make_axes_locatable(axes[x + epoch*5])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_ticks([0.025 - scaler, 0.025, 0.025 + scaler])
            cbar.set_ticklabels(['-' + str(round(scaler*10000)), '', str(round(scaler*10000))])
            cbar.ax.set_ylabel('Position Evidence (1e-4)', rotation=270,
                       fontsize=11, fontweight='bold')   
            
        axes[x + epoch*5].plot([21, 21],[0, 800],'k', linestyle = '--', linewidth = 3)
        axes[x + epoch*5].plot([21, 1],[0, 500],'k', linestyle = '--', linewidth = 3)
        axes[x + epoch*5].set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
        axes[x + epoch*5].set_ylim([1, 400])
        axes[x + epoch*5].set_xlim([0.5, 40.5])
        axes[x + epoch*5].xaxis.set_ticks_position('bottom')
        
        if epoch == 0: 
            axes[x + epoch*5].set_title(str(times[x]) + ' ms', 
                                        fontsize = 15, fontweight = 'bold')

axes[0].text(2, 220, "No compensation", rotation = -73, fontsize = 9)
axes[0].text(21.5, 210, "Full compensation", rotation = -90, fontsize = 9)

plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=3.5)
