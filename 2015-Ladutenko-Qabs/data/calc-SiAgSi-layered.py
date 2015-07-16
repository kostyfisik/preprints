#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#    Copyright (C) 2009-2015 Ovidio Peña Rodríguez <ovidio@bytesfall.com>
#    Copyright (C) 2013-2015  Konstantin Ladutenko <kostyfisik@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    The only additional remark is that we expect that all publications
#    describing work using this software, or all commercial products
#    using it, cite the following reference:
#    [1] O. Pena and U. Pal, "Scattering of electromagnetic radiation by
#        a multilayered sphere," Computer Physics Communications,
#        vol. 180, Nov. 2009, pp. 2348-2354.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import matplotlib.pyplot as plt
from scattnlay import scattnlay
from scattnlay import scattcoeffs
###############################################################################
def SetXM(design, extra_width = 0, WL=0, epsilon_Si=0, epsilon_Ag=0):
    """ design value:
    1: AgSi - a1
    2: SiAgSi - a1, b1
    3: SiAgSi - a1, b2
    """
    if WL == 0:
        epsilon_Si = 18.4631066585 + 0.6259727805j
        epsilon_Ag = -8.5014154589 + 0.7585845411j

    index_Si = np.sqrt(epsilon_Si)
    index_Ag = np.sqrt(epsilon_Ag)
    isSiAgSi=True
    isBulk = False
    if design==1:
        #36	5.62055	0	31.93	4.06	49	5.62055	500
        isSiAgSi=False
        if WL == 0: WL=500 #nm
        core_width = 0.0 #nm Si
        inner_width = 31.93 #nm Ag
        outer_width = 4.06 #nm  Si
    elif design==2:
        #62.5	4.48866	29.44	10.33	22.73	0	4.48866	500
        if WL == 0: WL=500 #nm
        core_width = 29.44 #nm Si
        inner_width = 10.33 #nm Ag
        outer_width = 22.73 #nm  Si
    elif design == 3:
        #81.4	3.14156	5.27	8.22	67.91	0	3.14156	500
        if WL == 0: WL=500 #nm
        core_width = 5.27 #nm Si
        inner_width = 8.22 #nm Ag
        outer_width = 67.91 #nm  Si

    elif design==4:
        if WL == 0: WL=800 #nm
        # WL=800 #nm
        # epsilon_Si = 13.64 + 0.047j
        # epsilon_Ag = -28.05 + 1.525j
        core_width = 17.74 #nm Si
        inner_width = 23.31 #nm Ag
        outer_width = 22.95 #nm  Si
    elif design==5:  # Bashevoj
        if WL == 0: WL=354 #nm
        core_r = WL/20.0
        epsilon_Ag = -2.0 + 0.28j   #original
        index_Ag = np.sqrt(epsilon_Ag)
        x = np.ones((1), dtype = np.float64)
        x[0] = 2.0*np.pi*core_r/WL
        m = np.ones((1), dtype = np.complex128)
        m[0] = index_Ag
        # x = np.ones((2), dtype = np.float64)
        # x[0] = 2.0*np.pi*core_r/WL/4.0*3.0
        # x[1] = 2.0*np.pi*core_r/WL
        # m = np.ones((2), dtype = np.complex128)
        # m[0] = index_Ag
        # m[1] = index_Ag
        return x, m, WL
    outer_width = outer_width+extra_width

    core_r = core_width
    inner_r = core_r+inner_width
    outer_r = inner_r+outer_width

    nm = 1.0
    if isSiAgSi:
        x = np.ones((3), dtype = np.float64)
        x[0] = 2.0*np.pi*core_r/WL
        x[1] = 2.0*np.pi*inner_r/WL
        x[2] = 2.0*np.pi*outer_r/WL
        m = np.ones((3), dtype = np.complex128)
        m[0] = index_Si/nm
        m[1] = index_Ag/nm
    #    m[0, 1] = index_Si/nm
        m[2] = index_Si/nm
    else:
        # bilayer
        x = np.ones((2), dtype = np.float64)
        x[0] = 2.0*np.pi*inner_r/WL
        x[1] = 2.0*np.pi*outer_r/WL
        m = np.ones((2), dtype = np.complex128)
        m[0] = index_Ag/nm
        m[1] = index_Si/nm
    return x, m
###############################################################################
def calc(design, extra_width, WL, epsilon_Si, epsilon_Ag):
    x, m = SetXM(design, extra_width, WL, epsilon_Si, epsilon_Ag)
    terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(
        np.array([x]), np.array([m]))
    terms2, an, bn = scattcoeffs(np.array([x]), np.array([m]), terms)
    an_s = (an.real - np.abs(an)**2)[0]
    bn_s = (bn.real - np.abs(bn)**2)[0]
    return np.array([WL, an_s[0], bn_s[0], an_s[1], bn_s[1], an_s[2], bn_s[2] ])
###############################################################################
def GetEpsilon(WLs, fname):
    data = np.loadtxt(fname)
    WL = data[:,0]
    epsRe = data[:,1]
    epsIm = data[:,2]
    from scipy.interpolate import interp1d
    fRe = interp1d(WL, epsRe)
    fIm = interp1d(WL, epsIm)
    # fRe = interp1d(WL, epsRe, kind=2)
    # fIm = interp1d(WL, epsIm, kind=2)

    data = np.vstack((WLs, fRe(WLs)+fIm(WLs)*1j))
    # data = np.concatenate(WLs, np.array(fRe(WLs)))
    # data = np.concatenate(WLs, )
    return np.transpose(data)
###############################################################################
def save_spectra(fname, from_WL, to_WL, total_points, design, extra_width):
    WLs = np.linspace(from_WL, to_WL, total_points)
    epsSi = GetEpsilon(WLs, "Si-int.txt")
    epsAg = GetEpsilon(WLs, "Ag-int.txt")
    data = calc(design, extra_width, WLs[0], epsSi[0,1], epsAg[0,1])
    for i in xrange(len(WLs)):
        data = np.vstack((data,calc(design, extra_width, WLs[i], epsSi[i,1], epsAg[i,1])))
    data = data[1:]
    np.savetxt(fname,data)
###############################################################################
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
###############################################################################

# design = 1 #AgSi
# # epsilon_Si = 18.4631066585 + 0.6259727805j
# # epsilon_Ag = -8.5014154589 + 0.7585845411j
# # WL = 500
from_WL = 400
to_WL = 600
total_points = 400
for i in xrange(1):
    extra_width = 0
    #fname = "sum-abs-spectra.dat"
    fname = "absorb-layered-spectra-d%i.dat"%extra_width 
    # design = 1 #AgSi
    # save_spectra(fname, from_WL, to_WL, total_points, design, extra_width)
    data, data_spaced = load_data(fname)

    #fname2 = "Si-Ag-Si-channels-TotalR063-calc.dat"
    fname2 = "Si-Ag-Si-channels-TotalR081-calc.dat"
    design = 3
    save_spectra(fname2, from_WL, to_WL, total_points, design, extra_width)
    data2, data_spaced2 = load_data(fname2)
    data_spaced2 = data2

    # fname3 = "Si-Ag-Si-channels-TotalR081-calc.dat"
    # design = 3
    # save_spectra(fname3, from_WL, to_WL, total_points, design, extra_width)
    # data3, data_spaced3 = load_data(fname3)
    # data_spaced3 = data3

    isAll = False
    #isAll = True
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
    # for i in xrange(len(data[0,:])-1):
    # for i in xrange(6):
    colors = ['b', 'r', 'g']
    for l in xrange(3):
        maxl = max(np.abs(data[:,l*6+1]))
        for n in xrange(6):
            maxl = max(maxl, max(np.abs(data[:,l*6+n+1])))
            #print (maxl)
        for n in xrange(1):
            style = '-'
            # cax = axs[AgSi].plot(data[:,0], np.abs( data[:,l*6+n*2+1]/max(np.abs(data[:,l*6+n*2+1]))),  linewidth=0.5+l*0.2, ls = style,
            #              label="e(%i,%i)"%(l,n), color=colors[n])
            cax = axs[AgSi].plot(data[:,0],  data[:,l*6+n*2+1],  linewidth=0.5+l*0.2, ls = style,
                         label="e(%i,%i)"%(l,n), color=colors[n])
            # style = '--'
            # cax = axs[AgSi].plot(data[:,0], np.abs( data[:,l*6+n*2+2]/max(np.abs(data[:,l*6+n*2+2]))),  linewidth=0.5+l*0.2, ls = style,
            #              label="m(%i,%i)"%(l,n), color=colors[n])

    # cax = axs[AgSi].plot(data[:,0], data[:,1], linewidth=plotwidth,
    #                      solid_joinstyle='round', solid_capstyle='round', color='black'
    #                      , label=r"$\tilde{a}_1$"
    # )
    # cax = axs[AgSi].plot(data[:,0], data[:,2], linewidth=plotwidth/1.5,
    #                      solid_joinstyle='round', solid_capstyle='round', color='red'
    #                      , label=r"$\tilde{b}_1$"
    # )
    # cax = axs[AgSi].plot(data[:,0], data[:,3], linewidth=plotwidth,
    #                      solid_joinstyle='round', solid_capstyle='round', color='green'
    #                      , label=r"$\tilde{a}_2$"
    # )
    # cax = axs[AgSi].plot(data[:,0], data[:,4], linewidth=plotwidth/1.5,
    #                      solid_joinstyle='round', solid_capstyle='round', color='blue'
    #                      , label=r"$\tilde{b}_2$"
    # )
    # axs[AgSi].axhline(y=0.25, ls='--', dashes=[2,2], color='gray')
    #lg=axs[AgSi].legend(loc='center right',prop={'size':11})
    #lg=axs[SiAgSi].legend(loc='upper right',prop={'size':8})
    lg=axs[AgSi].legend(loc='upper right',prop={'size':6})
    lg.get_frame().set_linewidth(0.0)
    # axs[AgSi].annotate('0.25', xy=(530, 0.25), fontsize=9, color='gray',
    #                 horizontalalignment='left', verticalalignment='bottom')

    lg.draw_frame(False)

    plotwidth=2.0
    cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,1], linewidth=plotwidth,
                         solid_joinstyle='round', solid_capstyle='round', color='black'
                         , label=r"$\tilde{a}_1$"
    )

    cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,2], linewidth=plotwidth/1.5,
                         solid_joinstyle='round', solid_capstyle='round', color='red'
                         , label=r"$\tilde{b}_1$"
    )
    cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,3], linewidth=plotwidth,
                         solid_joinstyle='round', solid_capstyle='round', color='green'
                         , label=r"$\tilde{a}_2$"
    )
    cax = axs[SiAgSi].plot(data_spaced2[:,0], data_spaced2[:,4], linewidth=plotwidth/1.5,
                         solid_joinstyle='round', solid_capstyle='round', color='blue'
                         , label=r"$\tilde{b}_2$"
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
                             , label=r"$\tilde{a}_1$"
        )

        cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,2], linewidth=plotwidth/1.5,
                             solid_joinstyle='round', solid_capstyle='round', color='red'
                             , label=r"$\tilde{b}_1$"
        )
        cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,3], linewidth=plotwidth,
                             solid_joinstyle='round', solid_capstyle='round', color='green'
                             , label=r"$\tilde{a}_2$"
        )
        cax = axs[SiAgSi2].plot(data_spaced3[:,0], data_spaced3[:,4], linewidth=plotwidth/1.5,
                             solid_joinstyle='round', solid_capstyle='round', color='blue'
                             , label=r"$\tilde{b}_2$"
        )
        axs[SiAgSi2].axhline(y=0.25, ls='--', dashes=[2,2], color='gray')
        lg=axs[SiAgSi2].legend(loc='center right',prop={'size':11})
        #lg=axs[SiSiAgSi2].legend(loc='upper right',prop={'size':8})
        #lg.get_frame().set_linewidth(0.0)
        axs[SiAgSi2].annotate('0.25', xy=(530, 0.25), fontsize=9, color='gray',
                        horizontalalignment='left', verticalalignment='bottom')

        lg.draw_frame(False)


    y_up_lim = 0.29
    #axs[AgSi].set_ylabel(r'$\tilde{a}_n^{(l)}+\tilde{d}_n^{(l)} ,\ \tilde{b}_n^{(l)}+\tilde{c}_n^{(l)}$', labelpad=-0.9)
    axs[AgSi].set_ylabel(r'${a}_{ln}^2+d_{ln}^2 ,\ b_{ln}^2+c_{ln}^2$', labelpad=-0.9)
    # axs[AgSi].set_ylim(0, y_up_lim)

    axs[SiAgSi].set_ylabel(r'$\tilde{a}_n ,\ \tilde{b}_n$', labelpad=-0.9)
    axs[SiAgSi].set_ylim(0, y_up_lim)

    if isAll:
        axs[SiAgSi2].set_ylabel(r'$\tilde{a}_n ,\ \tilde{b}_n$', labelpad=-0.9)
        axs[SiAgSi2].set_ylim(0, y_up_lim)
        axs[SiAgSi2].set_xlabel('Wavelengh, nm', labelpad=2)
    else:
        axs[SiAgSi].set_xlabel('Wavelengh, nm', labelpad=2)
    plt.xlim(from_WL,  to_WL)

    axs[AgSi].annotate(r'$\Delta=%i$'%extra_width, xy=(0.09, 0.985), xycoords='axes fraction', fontsize=10,
                    horizontalalignment='left', verticalalignment='top')

    axs[AgSi].set_yscale('log')
    axs[AgSi].annotate('(a)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                    horizontalalignment='right', verticalalignment='top')
    axs[SiAgSi].annotate('(b)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                    horizontalalignment='right', verticalalignment='top')
    if isAll:
        axs[SiAgSi2].annotate('(c)', xy=(0.99, 0.985), xycoords='axes fraction', fontsize=10,
                    horizontalalignment='right', verticalalignment='top')

    fig.subplots_adjust(hspace=.05)

    fname="2015-04-01-SiAgSi-ab-spectra-calc-d%+03i"%extra_width
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


