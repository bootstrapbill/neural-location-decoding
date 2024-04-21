# Author William Turner williamfrancisturner@gmail.com
# 

import os   
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from plot_funcs import weight_mean_circ, load_LDA, calc_FWHM

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

# average over motion directions and participants
start = np.mean(np.mean(allData[[0,3],:,:,:], axis = 0), axis = 2)
stop = np.mean(np.mean(allData[[1,4],:,:,:], axis = 0), axis = 2)
reversal = np.mean(np.mean(allData[[2,5],:,:,:], axis = 0), axis = 2)

###### FIG 2 ######

fig = plt.figure(figsize=(15, 8),dpi=300)
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)

times = np.linspace(-1000, 1000, 1024)
scaler = 0.0005
chance = 1/40

im = ax1.matshow(start, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto') 
ax1.axhline(0, color='k', linestyle='--', lw=3)
ax1.plot([40, 21], [475, 0], 'k-', lw=4)
ax1.plot([1, 21], [500, 1000], 'k-', lw=4)
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax1.set_yticks([-1000, 0, 1000])
ax1.set_ylabel('Testing Time (ms)', fontsize=20, fontweight='bold')
ax1.set_ylim((-1000, 1000))
ax1.set_xlim((0.5, 40.5))

im = ax2.matshow(stop, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm',
                   origin='lower', aspect = 'auto')  
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

im = ax3.matshow(reversal, extent=[0.5, 40.5, -1000, 1000],
                   vmin=chance - scaler, vmax=chance + scaler, cmap='coolwarm', 
                   origin='lower', aspect = 'auto')  
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

#### Figure 3 #######

colorCentroid = 'darkorange'
colorMax = 'green'

fig = plt.figure(figsize=(18, 12), dpi=300)
ax1 = plt.subplot(3,3,1)
ax2 = plt.subplot(3,3,2)
ax3 = plt.subplot(3,3,3)
ax4 = plt.subplot(2,2,3)
ax5 = plt.subplot(2,2,4)

## get array of stimulus positions over time
pos = np.linspace(1,41,512) 
pos[pos>40.5] -= 40
pos = np.hstack((pos[256:], pos[:256])) - 1
 
slices = np.linspace(38, 65, 27).astype(int) # ~75-125 ms 

alignedStart = np.zeros([len(slices), 40])

for x, i in enumerate(slices):
    
    position = int(np.round(pos[int(i)]))
    plotMap = np.concatenate((start[512 + i, position:], start[512 + i, :position]))
    alignedStart[x, :] = plotMap 

# centre data & average over time   
alignedStart = np.concatenate((alignedStart[:, 20:], alignedStart[:, :20]), axis = 1)
avg = np.mean(alignedStart, axis= 0)

# calculate centroid 
wa, mag = weight_mean_circ(avg)

xlocs = np.linspace(1,40,40)

# plot dummy data for legend
ax1.scatter(0, 0, color=colorMax, marker = "v", s = 100)
ax1.scatter(0, 0, color=colorCentroid, marker = "v", s = 100)
ax1.legend(['Peak', 'Centroid'], loc = 'upper right', fontsize = 15)

ax1.scatter(xlocs[np.argmax(avg)], 0.0261, color=colorMax, marker = "v", s = 200)
ax1.scatter(wa, 0.0259, color=colorCentroid, marker = "v", s = 200)
ax1.plot(xlocs, avg, color='grey', linestyle='-', lw=1)

calc_FWHM(avg) # this will print out FWHM to console/log

norm = plt.Normalize(chance - scaler, chance + scaler)
cmap = matplotlib.cm.get_cmap('coolwarm')
bar_colors = cmap(norm(avg))

ax1.bar(xlocs[avg >= chance], avg[avg >= chance]-chance, bottom = chance, color=cmap(norm(avg[avg>=chance])), width = 1)
ax1.bar(xlocs[avg < chance], abs(chance - avg[avg < chance]), bottom = avg[avg < chance], color=cmap(norm(avg[avg<chance])), width = 1)

ax1.set_xlim((0.5, 40.5))
ax1.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax1.set_title('Onset', fontweight='bold', fontsize = 25)
ax1.set_ylim((chance-0.0005, chance + 0.0012))
ax1.set_yticks([chance, chance + 0.0010], ['0', '10'])
ax1.set_ylabel('Evidence (1e-4)', fontsize=20, fontweight='bold')
ax1.axvline(21, color="black", linestyle='--', lw=1) 
ax1.vlines(21,chance-0.00045, chance-0.00035, color="black", linestyle='-', lw=10) 
ax1.axhline(chance, color='grey', linestyle='-', lw=1)
ax1.arrow(21, chance - 0.0004, 5, 0, color='black', width = 0.00003, head_width=0.0001, head_length=1)

onsetData = avg 

## get fixed view of lead up to stop

slices = np.linspace(0, 511, 512).astype(int) # ~ -1000-0 ms
alignedStop = np.zeros([len(slices), 40])

for x, i in enumerate(slices):
    
    position = int(np.round(pos[int(i)]))
    plotMap = np.concatenate((stop[i, position:], stop[i, :position]))
    alignedStop[x, :] = plotMap

alignedStop = np.concatenate((alignedStop[:, 20:], alignedStop[:, :20]), axis = 1)
avg = np.mean(alignedStop, axis= 0)

wa, mag = weight_mean_circ(avg)

xlocs = np.linspace(1,40,40)

cm = plt.get_cmap('coolwarm')
ax2.scatter(xlocs[np.argmax(avg)], 0.0261, color=colorMax, marker = "v", s = 200)
ax2.scatter(wa, 0.0259, color=colorCentroid, marker = "v", s = 200)
ax2.plot(xlocs,avg, color='grey', linestyle='-', lw=1)

calc_FWHM(avg) # this will print out FWHM to console/log

norm = plt.Normalize(chance - scaler, chance + scaler)
cmap = matplotlib.cm.get_cmap('coolwarm')
bar_colors = cmap(norm(avg))

ax2.bar(xlocs[avg >= chance], avg[avg >= chance]-chance, bottom = chance, color=cmap(norm(avg[avg>=chance])), width = 1)
ax2.bar(xlocs[avg < chance], abs(chance - avg[avg < chance]), bottom = avg[avg < chance], color=cmap(norm(avg[avg<chance])), width = 1)

ax2.set_xlim((0.5, 40.5))
ax2.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax2.set_title('Mid-motion', fontweight='bold', fontsize = 25)
ax2.set_ylim((chance-0.0005, chance + 0.0012))
ax2.set_yticks([chance, chance + 0.0010], ['0', '10'])
ax2.axvline(21, color="black", linestyle='--', lw=1) 
ax2.vlines(21,chance-0.00045, chance-0.00035, color="black", linestyle='-', lw=10) 
ax2.axhline(chance, color='grey', linestyle='-', lw=1)
ax2.arrow(21, chance - 0.0004, 5, 0, color='black', width = 0.00003, head_width=0.0001, head_length=1)

motionData = avg 

# calculate difference between motion and start
differenceData = motionData - onsetData

norm = plt.Normalize(0 - scaler, 0 + scaler)
cmap = matplotlib.cm.get_cmap('coolwarm')
bar_colors = cmap(norm(avg))

ax3.plot(np.linspace(1,40,40),differenceData, color='grey', linestyle='-', lw=1)
ax3.bar(xlocs[differenceData >= 0], differenceData[differenceData >= 0], bottom = 0, color=cmap(norm(differenceData[differenceData>=0])), width = 1)
ax3.bar(xlocs[differenceData < 0], abs(0 - differenceData[differenceData < 0]), bottom = differenceData[differenceData < 0], color=cmap(norm(differenceData[differenceData<0])), width = 1)

ax3.set_xlim((0.5, 40.5))
ax3.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax3.set_title('Difference', fontweight='bold', fontsize = 25)
ax3.set_ylim((0-0.0005, 0 + 0.0012))
ax3.set_yticks([0, 0 + 0.0010], ['0', '10'])
ax3.axvline(21, color="black", linestyle='--', lw=1) 
ax3.vlines(21,-0.00045, -0.00035, color="black", linestyle='-', lw=10) 
ax3.axhline(0, color='grey', linestyle='-', lw=1)
ax3.arrow(21, 0 - 0.0004, 5, 0, color='black', width = 0.00003, head_width=0.0001, head_length=1)

times = np.linspace(-1000, 1000, len(stop))
wa, mag = weight_mean_circ(stop)
size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 800
peak = np.argmax(stop, axis = 1) + 1

wa_down = wa[::8] # sample every ~15 ms
peak_down = peak[::8] 
size_down = size[::8] 
times_down = times[::8] 

ax4.plot([21, 40], [-1000, -525], 'k-', lw=3)
ax4.plot([1, 21], [-500, 0], 'k-', lw=3)
ax4.scatter(peak_down, times_down, s = size_down, c = colorMax, marker = ".", zorder = 10)
ax4.scatter(wa_down, times_down, c = colorCentroid, s = size_down, marker = ".", zorder = 10)
ax4.axhline(0., color='k', linestyle='--', lw=2)
ax4.xaxis.set_ticks_position('bottom')
ax4.set_xlim((0.5, 40.5))
ax4.set_yticks([-400, 0, 400])
ax4.set_ylim((-1000, 500))
ax4.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax4.set_title('Offset', fontweight='bold', fontsize = 25)
ax4.set_ylabel('Testing Time (ms)', fontsize=20, fontweight='bold')

wa, mag = weight_mean_circ(reversal)
size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 800
peak = np.argmax(reversal, axis = 1) + 1

wa_down = wa[::8] # sample every ~15 ms
peak_down = peak[::8] 
size_down = size[::8] 
 
ax5.scatter(0, 5000, color = colorMax, s = 100, marker = ".")
ax5.scatter(0, 5000, color = colorCentroid, s = 100, marker = ".") 
ax5.legend(['Peak', 'Centroid'], loc = 'upper right', fontsize = 20)
ax5.plot([21, 40], [-1000, -525], 'k-', lw=3)
ax5.plot([1, 21], [-500, 0], 'k-', lw=3)
ax5.plot([1, 21], [500, 0], 'k-', lw=3)
ax5.scatter(peak_down, times_down, s = size_down, c = colorMax, marker = ".", zorder = 10)
ax5.scatter(wa_down, times_down, c = colorCentroid, s = size_down, marker = ".", zorder = 10)
ax5.axhline(0., color='k', linestyle='--', lw=2)

ax5.xaxis.set_ticks_position('bottom')
ax5.set_ylim((-1000, 500))
ax5.set_xlim((0.5, 40.5))
ax5.set_yticks([])
ax5.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax5.set_title('Reversal', fontweight='bold', fontsize = 25)

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=-0.1)
fig.text(0.5, 0.04, 'Position', fontweight = 'bold', fontsize = 20, ha='center')
