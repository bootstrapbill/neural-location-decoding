# Author William Turner williamfrancisturner@gmail.com
# 

import os   
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from plot_funcs import weight_mean_circ, loadLDA, smooth_angles

os.chdir('/Users/willturner/Documents/Uni/Postdoc 2021/Decoding_Smooth_Reversals_EEG_data/data')

pIDs = ["01","02","03","04","05","06","07","08","09",
        "10","11","12","13","14","15","16","17","18"]

allData = loadLDA(pIDs, 'LDA')

# flip one motion direction 
allData[[3,4,5], :, :, :] = np.flip(allData[[3,4,5], :, :, :], axis = 2)

# average over motion directions
start = np.mean(np.mean(allData[[0,3],:,:,:], axis = 0), axis = 2)
stop = np.mean(np.mean(allData[[1,4],:,:,:], axis = 0), axis = 2)
reversal = np.mean(np.mean(allData[[2,5],:,:,:], axis = 0), axis = 2)

###### FIG 2 #########

fig = plt.figure(figsize=(15, 10),dpi=300)
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)

times = np.linspace(-1000, 1000, 1024)
scaler = 0.0003
chance = 1/40

im = ax1.matshow(start, extent=[1, 40, -1000, 1000],
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
ax1.set_title('Onset', fontsize=20, fontweight='bold')

im = ax2.matshow(stop, extent=[1, 40, -1000, 1000],
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
ax2.set_title('Offset', fontsize=20, fontweight='bold')

im = ax3.matshow(reversal, extent=[1, 40, -1000, 1000],
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
ax3.set_title('Reversal', fontsize=20, fontweight='bold')

matplotlib.rcParams.update({'font.size': 15})
cb_ax = fig.add_axes([1, 0.55, 0.02, 0.4])
cbar = fig.colorbar(im, cax=cb_ax)
cbar.set_ticks([chance - scaler, chance, chance + scaler])
cbar.set_ticklabels(['-' + str(round(scaler*10000)), '', str(round(scaler*10000))])
cbar.ax.set_ylabel('Position Evidence (1e-4)', rotation=270,
                   fontsize=14, fontweight='bold')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)

plt.savefig('Fig2.png')

##############################################################################

colorCentroid = 'green'
colorMax = 'purple'

fig = plt.figure(figsize=(15, 10), dpi=300)
ax4 = plt.subplot(2,2,1)
ax5 = plt.subplot(2,2,2)
ax6 = plt.subplot(2,2,3)
ax7 = plt.subplot(2,2,4)

## get array of stimulus positions over time
pos = np.linspace(1,41,512) 
pos[pos>40.5] -= 40
pos = np.hstack((pos[256:], pos[:256])) - 1 # subtract 1 to use as index
 
slices = np.linspace(39,65,27).astype(int) # ~75-125 ms 
alignedStart = np.zeros([len(slices), 40])

for x, i in enumerate(slices):
    
    position = int(np.round(pos[int(i)]))
    plotMap = np.concatenate((start[511 + i, position:], start[511 + i, :position]))
    alignedStart[x, :] = plotMap 

# centre data & average over time   
alignedStart = np.concatenate((alignedStart[:, 20:], alignedStart[:, :20]), axis = 1)
avg = np.mean(alignedStart, axis= 0)

# calculate centroid 
wa, mag = weight_mean_circ(avg)

xlocs = np.linspace(1,40,40)
ax4.scatter(np.argmax(avg) + 1, 0.0264, color=colorMax, marker = "v", s = 200)
ax4.scatter(wa, 0.0261, color=colorCentroid, marker = "v", s = 200)
ax4.plot(np.linspace(1,40,40), avg, color='grey', linestyle='-', lw=1)

# calculate estimate of FWHM: 
HM = (1/40) + ((avg.max() - (1/40))/2)
diffs = abs(avg-HM)
HM1 = np.argmin(diffs)
diffs[HM1] = 100
HM2 = np.argmin(diffs)
width = abs(HM1-HM2)
print('FWHM start ' + str(width * 9) + ' polar degrees')

norm = plt.Normalize(chance - scaler, chance + scaler)
cmap = matplotlib.cm.get_cmap('coolwarm')
bar_colors = cmap(norm(avg))

ax4.bar(xlocs[avg >= chance], avg[avg >= chance]-chance, bottom = chance, color=cmap(norm(avg[avg>=chance])), width = 1)
ax4.bar(xlocs[avg < chance], abs(chance - avg[avg < chance]), bottom = avg[avg < chance], color=cmap(norm(avg[avg<chance])), width = 1)

ax4.set_xlim((1, 40))
ax4.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax4.set_title('Onset', fontweight='bold', fontsize = 18)
ax4.set_ylim((chance-0.0005, chance + 0.0016))
ax4.set_yticks([chance, chance + 0.0010], ['0', '10'])
ax4.set_ylabel('Position Evidence (1e-4)', fontsize=18, fontweight='bold')
ax4.axvline(21, color="black", linestyle='-', lw=3) 
ax4.axhline(chance, color='grey', linestyle='-', lw=1)
ax4.arrow(21, chance + 0.0003, 5, 0, color='black', width = 0.00003, head_width=0.0001, head_length=1)

## get fixed view of lead up to stop
slices = np.linspace(0, 511, 512).astype(int) 
alignedStop = np.zeros([len(slices), 40])

for x, i in enumerate(slices):
    
    position = int(np.round(pos[int(i)]))
    plotMap = np.concatenate((stop[i, position:], stop[i, :position]))
    alignedStop[x, :] = plotMap

# recentre and average over time (-500, 0 ms pre stop)
alignedStop = np.concatenate((alignedStop[:, 20:], alignedStop[:, :20]), axis = 1)
avg = np.mean(alignedStop, axis= 0)

wa, mag = weight_mean_circ(avg)

xlocs = np.linspace(1,40,40)

cm = plt.get_cmap('coolwarm')
ax5.scatter(np.argmax(avg) + 1, 0.0264, color=colorMax, marker = "v", s = 200)
ax5.scatter(wa, 0.0261, color=colorCentroid, marker = "v", s = 200)
ax5.plot(np.linspace(1,40,40),avg, color='grey', linestyle='-', lw=1)

# calculate estimate of FWHM: 
HM = (1/40) + ((avg.max() - (1/40))/2)
diffs = abs(avg-HM)
HM1 = np.argmin(diffs)
diffs[HM1] = 100
HM2 = np.argmin(diffs)
width = abs(HM1-HM2)
print('FWHM stop ' + str(width * 9) + ' polar degrees')

norm = plt.Normalize(chance - scaler, chance + scaler)
cmap = matplotlib.cm.get_cmap('coolwarm')
bar_colors = cmap(norm(avg))

ax5.bar(xlocs[avg >= chance], avg[avg >= chance]-chance, bottom = chance, color=cmap(norm(avg[avg>=chance])), width = 1)
ax5.bar(xlocs[avg < chance], abs(chance - avg[avg < chance]), bottom = avg[avg < chance], color=cmap(norm(avg[avg<chance])), width = 1)

ax5.set_xlim((1, 40))
ax5.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax5.set_title('Mid-motion', fontweight='bold', fontsize = 18)
ax5.set_ylim((chance-0.0005, chance + 0.0016))
ax5.set_yticks([chance, chance + 0.0010], ['0', '10'])
ax5.axvline(21, color="black", linestyle='-', lw=3) 
ax5.axhline(chance, color='grey', linestyle='-', lw=1)
ax5.arrow(21, chance + 0.0003, 5, 0, color='black', width = 0.00003, head_width=0.0001, head_length=1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)

#######################################################################s########

#fig = plt.figure(figsize=(8, 3.5), dpi=300)
#ax6 = plt.subplot(1,2,1)
#ax7 = plt.subplot(1,2,2)

wa, mag = weight_mean_circ(stop)
size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 200
peak = np.argmax(stop, axis = 1) + 1

bin_size = 5

wa_down = smooth_angles(wa, bin_size)[1::bin_size]
peak_down = smooth_angles(peak, bin_size)[1::bin_size]
size_down = np.round(np.convolve(size, np.ones(3), "valid")/3)[1::bin_size]

wa_down = wa
peak_down = peak
size_down = size

times = np.linspace(-1000, 1000, len(wa_down))

ax6.scatter(peak_down, times, s = size_down, c = colorMax, marker = ".")
ax6.scatter(wa_down, times, c = colorCentroid, s = size_down, marker = ".")
ax6.axhline(0., color='k', linestyle='--', lw=2)
ax6.plot([21, 40], [-1000, -525], lw=2)
ax6.plot([1, 21], [-500, 0], 'k-', lw=2)
ax6.plot([21, 40], [0, 475], 'k.', lw=2)
ax6.xaxis.set_ticks_position('bottom')
ax6.set_xlim((1, 40))
ax6.set_yticks([-400, 0, 400])
ax6.set_ylim((-500, 500))
ax6.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax6.set_title('Offset', fontweight='bold', fontsize = 18)
ax6.set_ylabel('Testing Time (ms)', fontsize=18, fontweight='bold')

wa, mag = weight_mean_circ(reversal)
size = (mag-np.min(mag))/(np.max(mag)-np.min(mag)) * 200
peak = np.argmax(reversal, axis = 1) + 1

wa_down = smooth_angles(wa, bin_size)[1::bin_size]
peak_down = smooth_angles(peak, bin_size)[1::bin_size]
size_down = np.round(np.convolve(size, np.ones(3), "valid")/3)[1::bin_size]

wa_down = wa
peak_down = peak
size_down = size
    
ax7.scatter(0, 5000, color = colorMax, marker = "v")
ax7.scatter(0, 5000, color = colorCentroid, marker = "v") # TO DO: should take circ_mean here
ax7.legend(['max', 'centroid'], loc = 'upper right', fontsize = 15)
ax7.scatter(peak_down, times, s = size_down, c = colorMax, marker = ".")
ax7.scatter(wa_down, times, c = colorCentroid, s = size_down, marker = ".")
ax7.axhline(0., color='k', linestyle='--', lw=2)
ax7.plot([21, 40], [-1000, -525], 'k-', lw=2)
ax7.plot([1, 21], [-500, 0], 'k-', lw=2)
ax7.plot([1, 21], [500, 0], 'k-', lw=2)
ax7.plot([20.5, 40], [1000, 500], 'k-', lw=2)
ax7.xaxis.set_ticks_position('bottom')
ax7.set_ylim((-500, 500))
ax7.set_xlim((1, 40))
ax7.set_yticks([])
ax7.set_xticks([1, 21, 40], [r'$-\pi$', '0', r'$\pi$'])
ax7.set_title('Reversal', fontweight='bold', fontsize = 18)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)

plt.savefig('Fig3.png')

