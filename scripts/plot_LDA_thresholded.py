import os   
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from plot_funcs import load_LDA, one_sided_ttest

os.chdir('data')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

pIDs = ["01","02","03","04","05","06","07","08","09",
        "10","11","12","13","14","15","16","17","18"]

allData = load_LDA(pIDs, 'LDA')

# flip one motion direction
allData[[3,4,5], :, :, :] = np.flip(allData[[3,4,5], :, :, :], axis = 2)

# average over motion directions 
start = np.mean(allData[[0,3],:,:,:], axis = 0)
stop = np.mean(allData[[1,4],:,:,:], axis = 0) 
reversal = np.mean(allData[[2,5],:,:,:], axis = 0)

###### FIG 2 ######

fig = plt.figure(figsize=(15, 8),dpi=300)
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)

times = np.linspace(-1000, 1000, 1024)
scaler = 0.0005
chance = 1/40

# create a binary mask for significant areas
p_values = one_sided_ttest(start-chance)
sig_mask_high = p_values < 0.05
sig_mask_low = p_values < 0.01

start = np.mean(start, axis = 2)

im = ax1.matshow(start, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.35) 
start_high = start 
start_low = start 
start_high[~sig_mask_high] = np.nan 
start_low[~sig_mask_low] = np.nan 
im2 = ax1.matshow(start_high, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.5) 
im3 = ax1.matshow(start_low, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 1) 

ax1.axhline(0, color='k', linestyle='--', lw=3)
ax1.plot([40, 21], [475, 0], 'k-', lw=4)
ax1.plot([1, 21], [500, 1000], 'k-', lw=4)
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax1.set_yticks([-1000, 0, 1000])
ax1.set_ylabel('Testing Time (ms)', fontsize=20, fontweight='bold')
ax1.set_ylim((-1000, 1000))
ax1.set_xlim((0.5, 40.5))

# create a binary mask for significant areas
p_values = one_sided_ttest(stop-chance)
sig_mask_high = p_values < 0.05
sig_mask_low = p_values < 0.01

stop = np.mean(stop, axis = 2)

im = ax2.matshow(stop, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.35) 
stop_high = stop 
stop_low = stop 
stop_high[~sig_mask_high] = np.nan 
stop_low[~sig_mask_low] = np.nan 
im2 = ax2.matshow(stop_high, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.5) 
im3 = ax2.matshow(stop_low, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 1) 

ax2.axhline(0, color='k', linestyle='--', lw=3)
ax2.plot([21, 40], [-1000, -525], 'k-', lw=4)
ax2.plot([1, 21], [-500, 0], 'k-', lw=4)
ax2.plot([21, 40], [0, 475], 'k.', lw=4)
ax2.xaxis.set_ticks_position('bottom')
ax2.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax2.set_xlabel('Position', fontsize = 20, fontweight = 'bold')
ax2.set_ylim((-1000, 1000))
ax2.set_yticks([])
ax2.set_xlim((0.5, 40.5))

# create a binary mask for significant areas
p_values = one_sided_ttest(reversal-chance)
sig_mask_high = p_values < 0.05
sig_mask_low = p_values < 0.01

reversal = np.mean(reversal, axis = 2)

im = ax3.matshow(reversal, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.35) 
reversal_high = reversal 
reversal_low = reversal 
reversal_high[~sig_mask_high] = np.nan 
reversal_low[~sig_mask_low] = np.nan 
im2 = ax3.matshow(reversal_high, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 0.5) 
im3 = ax3.matshow(reversal_low, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto', alpha = 1) 

ax3.axhline(0, color='k', linestyle='--', lw=3)
ax3.plot([21, 40], [-1000, -525], 'k-', lw=4)
ax3.plot([1, 21], [-500, 0], 'k-', lw=4)
ax3.plot([1, 21], [500, 0], 'k-', lw=4)
ax3.plot([21, 40], [1000, 525], 'k-', lw=4)
ax3.xaxis.set_ticks_position('bottom')
ax3.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax3.set_ylim((-1000, 1000))
ax3.set_yticks([])
ax3.set_xlim((0.5, 40.5))

matplotlib.rcParams.update({'font.size': 15})
cb_ax = fig.add_axes([1, 0.15, 0.02, 0.75])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks([chance - scaler, chance, chance + scaler])
cbar.set_ticklabels(['-' + str(round(scaler*10000)), '', str(round(scaler*10000))])
cbar.ax.set_ylabel('Position Evidence (1e-4)', rotation=270,
                   fontsize=14, fontweight='bold')

plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)