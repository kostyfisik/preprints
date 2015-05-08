#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def load_data(fname):
    data = np.loadtxt(fname)
    space = [np.nan]*len(data[0,:])
    #space = [-100]*len(data[0,:])
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    range_value = max_value-min_value
    max_step = range_value/20.0
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
    return data, data_spaced

fname = "Ag-Si-channels-TotalR036.dat"
data, data_spaced = load_data(fname)
for i in xrange(1, len(data[:,1])):
    if data[i-2,1]<=data[i,1] and data[i+2,1]<=data[i,1]:
        print(data[i,:])

fname2 = "Si-Ag-Si-channels-TotalR063.dat"
data2, data_spaced2 = load_data(fname2)

fname3 = "Si-Ag-Si-channels-TotalR081.dat"
data3, data_spaced3 = load_data(fname3)

isAll = False
isAll = True
############################# Plotting ######################
import numpy.ma as ma
vals = ma.array(data_spaced)
mvals = ma.masked_where(np.nan in data_spaced, vals)

if isAll:
    fig, axs = plt.subplots(3,figsize=(4,6), sharex=True)#, sharey=True)
else:
    fig, axs = plt.subplots(2,figsize=(4,4), sharex=True)#, sharey=True)
AgSi=0
SiAgSi=1
SiAgSi2=2
for ax in axs:
    ax.locator_params(axis='y',nbins=4)
    # for label in ['left', 'right', 'top', 'bottom']:
    #     ax.spines[label].set_position(('outward',-1.3))
    #ax.tick_params(axis='x', pad=30)

plotwidth=2.0
cax = axs[AgSi].plot(data[:,0], data[:,1], linewidth=plotwidth,
                     solid_joinstyle='round', solid_capstyle='round', color='black'
                     , label=r"$a_1$"
)
cax = axs[AgSi].plot(data[:,0], data[:,2], linewidth=plotwidth/1.5,
                     solid_joinstyle='round', solid_capstyle='round', color='red'
                     , label=r"$b_1$"
)
cax = axs[AgSi].plot(data[:,0], data[:,3], linewidth=plotwidth,
                     solid_joinstyle='round', solid_capstyle='round', color='green'
                     , label=r"$a_2$"
)
cax = axs[AgSi].plot(data[:,0], data[:,4], linewidth=plotwidth/1.5,
                     solid_joinstyle='round', solid_capstyle='round', color='blue'
                     , label=r"$b_2$"
)
axs[AgSi].axhline(y=0.25, ls='--', dashes=[2,2], color='gray')
lg=axs[AgSi].legend(loc='center right',prop={'size':11})
#lg=axs[SiAgSi].legend(loc='upper right',prop={'size':8})
#lg.get_frame().set_linewidth(0.0)
axs[AgSi].annotate('0.25', xy=(530, 0.25), fontsize=9, color='gray',
                horizontalalignment='left', verticalalignment='bottom')

lg.draw_frame(False)

plotwidth=2.0
cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,1], linewidth=plotwidth,
                     solid_joinstyle='round', solid_capstyle='round', color='black'
                     , label=r"$a_1$"
)

cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,2], linewidth=plotwidth/1.5,
                     solid_joinstyle='round', solid_capstyle='round', color='red'
                     , label=r"$b_1$"
)
cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,3], linewidth=plotwidth,
                     solid_joinstyle='round', solid_capstyle='round', color='green'
                     , label=r"$a_2$"
)
cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,4], linewidth=plotwidth/1.5,
                     solid_joinstyle='round', solid_capstyle='round', color='blue'
                     , label=r"$b_2$"
)
axs[SiAgSi].axhline(y=0.25, ls='--', dashes=[2,2], color='gray')
lg=axs[SiAgSi].legend(loc='center right',prop={'size':11})
#lg=axs[SiSiAgSi].legend(loc='upper right',prop={'size':8})
#lg.get_frame().set_linewidth(0.0)
axs[SiAgSi].annotate('0.25', xy=(530, 0.25), fontsize=9, color='gray',
                horizontalalignment='left', verticalalignment='bottom')

lg.draw_frame(False)

if isAll:
    plotwidth=2.0
    cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,1], linewidth=plotwidth,
                         solid_joinstyle='round', solid_capstyle='round', color='black'
                         , label=r"$a_1$"
    )

    cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,2], linewidth=plotwidth/1.5,
                         solid_joinstyle='round', solid_capstyle='round', color='red'
                         , label=r"$b_1$"
    )
    cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,3], linewidth=plotwidth,
                         solid_joinstyle='round', solid_capstyle='round', color='green'
                         , label=r"$a_2$"
    )
    cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,4], linewidth=plotwidth/1.5,
                         solid_joinstyle='round', solid_capstyle='round', color='blue'
                         , label=r"$b_2$"
    )
    axs[SiAgSi2].axhline(y=0.25, ls='--', dashes=[2,2], color='gray')
    lg=axs[SiAgSi2].legend(loc='center right',prop={'size':11})
    #lg=axs[SiSiAgSi2].legend(loc='upper right',prop={'size':8})
    #lg.get_frame().set_linewidth(0.0)
    axs[SiAgSi2].annotate('0.25', xy=(530, 0.25), fontsize=9, color='gray',
                    horizontalalignment='left', verticalalignment='bottom')

    lg.draw_frame(False)



y_up_lim = 0.29
axs[AgSi].set_ylabel(r'$a_n ,\ b_n$', labelpad=-0.9)
axs[AgSi].set_ylim(0, y_up_lim)

axs[SiAgSi].set_ylabel(r'$a_n ,\ b_n$', labelpad=-0.9)
axs[SiAgSi].set_ylim(0, y_up_lim)

if isAll:
    axs[SiAgSi2].set_ylabel(r'$a_n ,\ b_n$', labelpad=-0.9)
    axs[SiAgSi2].set_ylim(0, y_up_lim)
    axs[SiAgSi2].set_xlabel('Wavelengh, nm', labelpad=2)
else:
    axs[SiAgSi].set_xlabel('Wavelengh, nm', labelpad=2)
plt.xlim(400,  600)
axs[AgSi].annotate('(a)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='top')
axs[SiAgSi].annotate('(b)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='top')
if isAll:
    axs[SiAgSi2].annotate('(c)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='top')

fig.subplots_adjust(hspace=.05)

fname="2015-04-01-SiAgSi-ab-spectra"
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


