#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
fname = "2015-04-01-Qabs-SiAgSi-overview"
data = np.loadtxt(fname+".txt")
space = [np.nan]*len(data[0,:])
#space = [-100]*len(data[0,:])
min_value = np.min(data, axis=0)
max_value = np.max(data, axis=0)
range_value = max_value-min_value
print(min_value)
print(max_value)
print(range_value)
max_step = range_value/20.0
print(max_step)
data_spaced = data[0:1,:]
for i in xrange(2, len(data[:,0])+1):
    diff = np.absolute(data[i-1:i,:] - data[i-2:i-1,:])
    need_space = False
    for j in xrange(2,len(max_step)):
        if j > 4:
            continue
        if max_step[j]<diff[0,j]:
            need_space = True
    if need_space:
        data_spaced = np.concatenate((data_spaced,[space]))
    data_spaced = np.concatenate((data_spaced,data[i-1:i,:]))


import numpy.ma as ma
vals = ma.array(data_spaced)
mvals = ma.masked_where(np.nan in data_spaced, vals)
print(mvals)
# for i in xrange(0, len(data[:,0])):
#     print(mvals[i])


fig, axs = plt.subplots(3,figsize=(4,6), sharex=True)#, sharey=True)
NACS=0
Qsca=1
Design=2
for ax in axs:
    ax.locator_params(axis='y',nbins=4)
    # for label in ['left', 'right', 'top', 'bottom']:
    #     ax.spines[label].set_position(('outward',-1.3))
    #ax.tick_params(axis='x', pad=30)

plotwidth=2.0
cax = axs[NACS].plot(data_spaced[:,0], data_spaced[:,2], linewidth=plotwidth, solid_joinstyle='round', solid_capstyle='round', color='blue')
cax = axs[Qsca].plot(data_spaced[:,0], data_spaced[:,1], linewidth=plotwidth, solid_joinstyle='round', solid_capstyle='round', color='black')
cax = axs[Design].plot(data_spaced[:,0], data_spaced[:,2], linewidth=plotwidth, solid_joinstyle='round', solid_capstyle='round', color='red')
cax = axs[Design].plot(data_spaced[:,0], data_spaced[:,3], linewidth=plotwidth, solid_joinstyle='round', solid_capstyle='round', color='green')
cax = axs[Design].plot(data_spaced[:,0], data_spaced[:,4], linewidth=plotwidth, solid_joinstyle='round', solid_capstyle='round', color='blue')

axs[Qsca].set_ylabel('Qabs', labelpad=8.8)
axs[Qsca].set_ylim(0, 7)
axs[Design].set_ylabel('Width, nm', labelpad=2)
axs[Design].set_ylim(0, 75)
axs[Design].set_xlabel('Total R, nm', labelpad=2)
plt.xlim(0,  89)
fig.subplots_adjust(hspace=.05)

plt.savefig(fname+".pdf",pad_inches=0.02, bbox_inches='tight')
#plt.draw()

#plt.show()

plt.clf()
plt.close()

# cax = axs[0,0].imshow(Eabs_data, interpolation = 'nearest', cmap = cm.jet,
#                       origin = 'lower'
#                       #, vmin = min_tick, vmax = max_tick
#                       , extent = (min(scale_x), max(scale_x), min(scale_z), max(scale_z))
#                       #,norm = LogNorm()
#                       )


