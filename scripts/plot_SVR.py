# Author William Turner williamfrancisturner@gmail.com
# 

import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Wedge, Rectangle
import matplotlib
from plot_funcs import load_data, cluster_correct

os.chdir('data')

pIDs = ["01","02","03","04","05","06","07","08","09",
        "10","11","12","13","14","15","16","17","18"]

alpha = 0.01 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

#### Load data 

diag = load_data(pIDs, 'diag')
pos = load_data(pIDs, 'pos')
tgm = load_data(pIDs, 'tgm')
search = load_data(pIDs, 'search')
tf = load_data(pIDs, 'tf')
stacked = load_data(pIDs, 'stacked')

# Rescale scores to range [-1, 1]
# Note, this fixes an issue with the calculation of the 
# decoding scores which were accidentally scaled between -0.5 and 0.5
# rather than -1 and 1 as originally intended. See decoding_funcs.py
diag = 2 * (diag)
pos = 2 * (pos)
tgm = 2 * (tgm)
search = 2 * (search) 
tf = 2 * (tf)
stacked = 2 * (stacked) 

#### FIGURE 1 ####

fig = plt.figure(figsize=(19, 9))
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2, projection='polar')
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,(5,6))
ax = [ax1,ax2,ax3,ax4,ax5]

#### diagonal decoding

corrected_pvals = cluster_correct(diag)
sig_marker = corrected_pvals<alpha

se = np.std(diag, axis = 0, ddof = 1)/np.sqrt(diag.shape[0])
accuracy = np.mean(diag, axis=0)
time = np.linspace(-200, 1000, 615)

ax[0].fill_between(time, accuracy - se, accuracy + se, alpha=0.5, color="grey")
ax[0].plot(time, accuracy, color="black")
ax[0].scatter(time, -0.01*np.ones_like(time), c="black", alpha=sig_marker)
ax[0].set_xlim((-200, 1000))
ax[0].set_ylim((-0.02, 0.1))
ax[0].set_yticks(ticks=[0, 0.05, 0.1], labels=['0', '0.05', '0.1'])
ax[0].set_xticks(ticks=[0, 400, 800], labels=['0', '400', '800'])
ax[0].hlines(0, -200, 1000, colors="black")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_xlabel('Time (ms)', fontweight='bold')
ax[0].set_ylabel('Decoding Score', fontweight='bold')

#### plot position-specific decoding 

se = np.std(pos, axis = 0, ddof = 1)/np.sqrt(pos.shape[0])
accuracy = np.mean(pos, axis=0)
angles = -np.linspace(9,360,40) + 90 # angles drawn in reverse direciton in matlab
                                     # + 90 means pos 1 starts just right of vertical 

angles = np.append(angles, angles[0]) # make it fully wrap circle
se = np.append(se, se[0]) # same as above
accuracy = np.append(accuracy, accuracy[0]) # same as above

ax[1].fill_between(np.deg2rad(angles), accuracy - se, accuracy + se, alpha=0.5, color="grey")
ax[1].plot(np.deg2rad(angles), accuracy, color="black")
ax[1].set_yticks(ticks=[0, 0.05, 0.1], labels=['', '', ''])
ax[1].set_xticks(ticks=[0, np.pi/4, np.pi/2, (3/4)*np.pi, 
                        np.pi, (5/4) * np.pi, (3/2) * np.pi, (7/4) * np.pi], 
                 labels=['', '', '', '', '', '', '', ''])
ax[1].set_ylim((0, 0.12))

rect = ax[1].get_position()
rect = (rect.xmin-0.035, rect.ymin+rect.height/2 + 0.055, # x, y
         rect.width, rect.height/2) # width, height
scale_ax = ax[1].figure.add_axes(rect)
# hide most elements of the new axes
for loc in ['right', 'top', 'bottom']:
    scale_ax.spines[loc].set_visible(False)
scale_ax.tick_params(bottom=False, labelbottom=False)
scale_ax.patch.set_visible(False) # hide white background
scale_ax.spines['left'].set_bounds(*ax[1].get_ylim())
scale_ax.set_yticks(ax[1].get_yticks())
scale_ax.set_ylim(ax[1].get_rorigin(), ax[1].get_rmax())
scale_ax.set_ylabel('Decoding Score', fontsize = 16, fontweight = 'bold')

#### plot TGM
scaler = 0.05

corrected_pvals = cluster_correct(tgm)
sig_marker = corrected_pvals<alpha

data = np.mean(tgm, axis = 0)

im = ax[2].matshow(data, extent=[-200, 1000, -200, 1000],
                   vmin=-scaler, vmax=scaler, cmap='Greys_r', origin='lower', 
                   aspect = 'auto', alpha = 0.5)  
data[~sig_marker.T] = np.nan 
im = ax[2].matshow(data, extent=[-200, 1000, -200, 1000],
                   vmin=-scaler, vmax=scaler, cmap='coolwarm', origin='lower', 
                   aspect = 'auto')    
ax[2].hlines(0, -200, 1000, colors="black")
ax[2].vlines(0, -200, 1000, colors="black")
ax[2].set_xticks(ticks=[0, 400, 800], labels=['0', '400', '800'])
ax[2].set_yticks(ticks=[0, 400, 800], labels=['0', '400', '800'])
ax[2].tick_params(axis='x', bottom=True, labelbottom=True, labeltop=False)
ax[2].set_xlabel('Testing Time (ms)', fontweight='bold') 
ax[2].set_ylabel('Training Time (ms)', fontweight='bold') 
ax[2].set_ylim((-200, 1000))
ax[2].set_xlim((-200, 1000))

ax[2].hlines(75, 75, 125, colors="black")
ax[2].vlines(75, 75, 125, colors="black")
ax[2].hlines(125, 75, 125, colors="black")
ax[2].vlines(125, 75, 125, colors="black")

cbar = plt.colorbar(im, ax = ax[2], fraction=0.046, pad=0.04, ticks = [-scaler, 0, scaler])
cbar.ax.set_yticklabels(['-'+str(scaler), '', str(scaler)], va="center") 
cbar.ax.set_ylabel('Decoding Score', fontweight = 'bold')

#### plot frequency-specific decoding
corrected_pvals = cluster_correct(tf)
sig_marker = corrected_pvals<alpha

data = np.mean(tf, axis = 0)

im = ax[3].matshow(data, extent=[-200, 1000, 1, 41],
                vmin=-scaler, vmax=scaler, cmap='Greys_r',origin='lower',
                aspect='auto', alpha = 0.5)    
data[~sig_marker.T] = np.nan 
im = ax[3].matshow(data, extent=[-200, 1000, 1, 41],
                vmin=-scaler, vmax=scaler, cmap='coolwarm',origin='lower',
                aspect='auto')    
ax[3].set_xlabel('Time (ms)', fontweight='bold')
ax[3].set_ylabel('Freq (Hz)', fontweight='bold')
ax[3].set_yticks(ticks=[10, 20, 30, 40])
ax[3].set_xticks(ticks=[0, 400, 800], labels=['0', '400', '800'])
ax[3].tick_params(axis='x', bottom=True, labelbottom=True, labeltop=False)
ax[3].spines[['right', 'top']].set_visible(False)
ax[3].set_ylim((1, 41))
ax[3].set_xlim((-200, 1000))

#### plot stacked TGMs for successive stimuli
colors = ['Reds','Oranges','Greens','Blues','Purples']
timeShift = 0

for stim in [0,1,2,3,4]:
    
    plotData = stacked[:, stim, :, :]
    
    corrected_pvals = cluster_correct(plotData)
    sig_marker = corrected_pvals<alpha

    plotData = np.mean(plotData, axis = 0)
    plotData[~sig_marker.T] = np.nan
    # plotData[plotData < 0.01] = np.nan
    ax[4].matshow(plotData, extent=[-200 + timeShift, 500 + timeShift, 0, 500],
                       vmin=-scaler, vmax=scaler, cmap=colors[stim], origin='lower', aspect = 'auto')
    cmap = get_cmap(colors[stim])
    ax[4].vlines(timeShift, -200, 500, colors=cmap(0.99), linewidth = 2)
    ax[4].text(timeShift + 10, 465, 'Stim N+' + str(stim) if stim > 0 else 'Stim N', c=cmap(0.99), fontsize = 12)
    ax[4].set_ylim((0, 500))
    ax[4].set_xlim((-200, 1300))
    timeShift += 200

ax[4].set_xticks(ticks=[0, 400, 800, 1200], labels=['0', '400', '800', '1200'])
ax[4].set_yticks(ticks=[0, 200, 400], labels=['0', '200', '400'])
ax[4].tick_params(axis='x', bottom=True, labelbottom=True, labeltop=False)
ax[4].set_xlabel('Testing Time (ms)', fontweight='bold') 
ax[4].set_ylabel('Training Time (ms)', fontweight='bold')

#### plot searchlight
montage = mne.channels.make_standard_montage('biosemi64')
tempInfo = mne.create_info(montage.ch_names, 512, ch_types='eeg')
tempInfo.set_montage(montage)
picks = np.isin(tempInfo.ch_names, ['P7', 'P5', 'P3', 'P1', 'Pz', 
                                    'P2', 'P4', 'P6', 'P8', 'PO7',
                                    'PO3', 'POz', 'PO4', 'PO8',
                                    'O1', 'Oz', 'O2'])

ins = ax[0].inset_axes([0.6, 0.65, 0.3, 0.4])
dataTemp = np.mean(search, axis = 0)
mne.viz.plot_topomap(dataTemp, tempInfo,
                     axes=ins, cmap='coolwarm', 
                     contours = 0, mask = picks, 
                     vlim=(-scaler, scaler), res=5000)

plt.tight_layout(h_pad = 0.99)

os.chdir('figures')
plt.savefig('Fig1.png')


#### Plot Figure 1A (Tiling of localizer positions)
fig, ax = plt.subplots(figsize=(10,10))

flashLocs = np.linspace(9,360,40)

a = Rectangle((-2,-2),4,4,facecolor = [0.5, 0.5, 0.5], edgecolor = "black")
ax.add_patch(a)

for loc in flashLocs: 
    
    a = Wedge(0, 1, loc-4.5, loc+4.5, width=0.2,facecolor="white",edgecolor="black")
    ax.add_patch(a)

ax.scatter(0,0,s=15,c="black")
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.axis('off')
plt.show()

plt.savefig('Fig1A.png')
