#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#    Copyright (C) 2009-2015 Ovidio Peña Rodríguez <ovidio@bytesfall.com>
#
#    This file is part of python-scattnlay
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

# This test case calculates the electric field in the 
# E-k plane, for an spherical Si-Ag-Si nanoparticle.

import scattnlay
from scattnlay import fieldnlay
from scattnlay import scattnlay
import numpy as np
import cmath


def get_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle
###############################################################################
def GetFlow3D(x0, y0, z0, max_length, max_angle, x, m):
    # Initial position
    flow_x = [x0]
    flow_y = [y0]
    flow_z = [z0]
    max_step = max(x0, y0, z0, x[-1])/50
    min_step = x[0]/2000
    step = min_step*3.0
    if max_step < min_step:
        max_step = min_step
    coord = np.vstack(([flow_x[-1]], [flow_y[-1]], [flow_z[-1]])).transpose()
    terms, E, H = fieldnlay(np.array([x]), np.array([m]), coord)
    Ec, Hc = E[0, 0, :], H[0, 0, :]
    S = np.cross(Ec, Hc.conjugate())
    Snorm = S/np.linalg.norm(S)
    Snorm_prev = Snorm.real
    length = 0
    dpos = step
    while length < max_length:
        step = step*2.0
        while step > min_step:
            #Evaluate displacement from previous poynting vector
            dpos = step
            dx = dpos*Snorm_prev[0];
            dy = dpos*Snorm_prev[1];
            dz = dpos*Snorm_prev[2];
            #Test the next position not to turn more than max_angle
            coord = np.vstack(([flow_x[-1]+dx], [flow_y[-1]+dy], [flow_z[-1]+dz])).transpose()
            terms, E, H = fieldnlay(np.array([x]), np.array([m]), coord)
            Ec, Hc = E[0, 0, :], H[0, 0, :]
            Eth = max(np.absolute(Ec))/1e10
            Hth = max(np.absolute(Hc))/1e10
            for i in xrange(0,len(Ec)):
                if abs(Ec[i]) < Eth:
                    Ec[i] = 0+0j
                if abs(Hc[i]) < Hth:
                    Hc[i] = 0+0j
            S = np.cross(Ec, Hc.conjugate())
            Snorm = S/np.linalg.norm(S)
            Snorm = Snorm.real
            angle = angle_between(Snorm, Snorm_prev)
            if angle < max_angle:
                break
            step = step/2.0
        #3. Save result
        Snorm_prev = Snorm
        dx = dpos*Snorm_prev[0];
        dy = dpos*Snorm_prev[1];
        dz = dpos*Snorm_prev[2];
        length = length + step
        flow_x.append(flow_x[-1] + dx)
        flow_y.append(flow_y[-1] + dy)
        flow_z.append(flow_z[-1] + dz)

    return np.array(flow_x), np.array(flow_y), np.array(flow_z)

###############################################################################
def GetFlow(scale_x, scale_z, Ec, Hc, a, b, npts, nmax):
    # Initial position
    flow_x = [a]
    flow_z = [b]
    x_pos = flow_x[-1]
    z_pos = flow_z[-1]
    x_idx = get_index(scale_x, x_pos)
    z_idx = get_index(scale_z, z_pos)
    S=np.cross(Ec[npts*z_idx+x_idx], Hc[npts*z_idx+x_idx]).real
    #if (np.linalg.norm(S)> 1e-4):
    Snorm_prev=S/np.linalg.norm(S)
    Snorm_prev=Snorm_prev.real
    max_x = np.max(scale_x)
    max_z = np.max(scale_z)
    min_x = np.min(scale_x)
    min_z = np.min(scale_z)
    for n in range(0, nmax):
        #Get the next position
        #1. Find Poynting vector and normalize it
        x_pos = flow_x[-1]
        z_pos = flow_z[-1]
        x_idx = get_index(scale_x, x_pos)
        z_idx = get_index(scale_z, z_pos)
        Epoint = Ec[npts*z_idx+x_idx]
        Hpoint = Hc[npts*z_idx+x_idx]
        S=np.cross(Epoint, Hpoint.conjugate())
        #if (np.linalg.norm(S)> 1e-4):
        Snorm=S.real/np.linalg.norm(S)
        #Snorm=Snorm.real
        #2. Evaluate displacement = half of the discrete and new position
        dpos = abs(scale_z[0]-scale_z[1])/2.0
        dx = dpos*Snorm[0];
        dz = dpos*Snorm[2];
        x_pos = x_pos+dx
        z_pos = z_pos+dz
        #3. Save result
        flow_x.append(x_pos)
        flow_z.append(z_pos)
        if x_pos<min_x or x_pos>max_x:
            break
        if z_pos<min_z or z_pos>max_z:
            break
    return flow_x, flow_z


###############################################################################
def SetXM(design):
    """ design value:
    1: AgSi - a1
    2: SiAgSi - a1, b1
    3: SiAgSi - a1, b2
    """
    epsilon_Si = 18.4631066585 + 0.6259727805j
    epsilon_Ag = -8.5014154589 + 0.7585845411j
    index_Si = np.sqrt(epsilon_Si)
    index_Ag = np.sqrt(epsilon_Ag)
    isSiAgSi=True
    if design==1:
        #36	5.62055	0	31.93	4.06	49	5.62055	500
        isSiAgSi=False
        WL=500 #nm
        core_width = 0.0 #nm Si
        inner_width = 31.93 #nm Ag
        outer_width = 4.06 #nm  Si
    elif design==2:
        #62.5	4.48866	29.44	10.33	22.73	0	4.48866	500
        WL=500 #nm
        core_width = 29.44 #nm Si
        inner_width = 10.33 #nm Ag
        outer_width = 22.73 #nm  Si
    else:
        #81.4	3.14156	5.27	8.22	67.91	0	3.14156	500
        WL=500 #nm
        core_width = 5.27 #nm Si
        inner_width = 8.22 #nm Ag
        outer_width = 67.91 #nm  Si
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
    return x, m, WL


###############################################################################
def GetField(crossplane, npts, factor, x, m):
    """
    crossplane: XZ, YZ, XY
    npts: number of point in each direction
    factor: ratio of plotting size to outer size of the particle
    x: size parameters for particle layers
    m: relative index values for particle layers
    """
    scan = np.linspace(-factor*x[-1], factor*x[-1], npts)
    coordX, coordZ = np.meshgrid(scan, scan)
    coordX.resize(npts*npts)
    coordZ.resize(npts*npts)
    coordY = coordX
    coordPlot = coordX
    zero = np.zeros(npts*npts, dtype = np.float64)
    if crossplane=='XY':
        coordZ = zero
    elif crossplane=='YZ':
        coordX = zero
    else:
        coordY = zero
    coord = np.vstack((coordX, coordY, coordZ)).transpose()
    terms, E, H = fieldnlay(np.array([x]), np.array([m]), coord)
    Ec = E[0, :, :]
    Hc = H[0, :, :]
    P=[]
    for n in range(0, len(E[0])):
        P.append(np.linalg.norm( np.cross(Ec[n], np.conjugate(Hc[n]) ).real/2 ))
    return Ec, Hc, P, coordPlot, coordPlot

###############################################################################
#design = 1
#design = 2
design = 3
x, m, WL = SetXM(design)
print "x =", x
print "m =", m
npts = 551
factor=2.8
crossplane='XZ'
crossplane='YZ'
Ec, Hc, P, coordX, coordZ = GetField(crossplane, npts, factor, x, m)

Er = np.absolute(Ec)
Hr = np.absolute(Hc)

# |E|/|Eo|
Eabs = np.sqrt(Er[ :, 0]**2 + Er[ :, 1]**2 + Er[ :, 2]**2)
Eangle = np.angle(Ec[ :, 0])/np.pi*180
Habs= np.sqrt(Hr[ :, 0]**2 + Hr[ :, 1]**2 + Hr[ :, 2]**2)
Hangle = np.angle(Hc[ :, 1])/np.pi*180

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LogNorm

    min_tick = 0.0
    max_tick = 0.00001

    Eabs_data = np.resize(P, (npts, npts)).T
    #Eabs_data = np.resize(Pabs, (npts, npts)).T
    # Eangle_data = np.resize(Eangle, (npts, npts)).T
    # Habs_data = np.resize(Habs, (npts, npts)).T
    # Hangle_data = np.resize(Hangle, (npts, npts)).T

    fig, ax = plt.subplots(1,1)#, sharey=True, sharex=True)
    #fig.tight_layout()
    # Rescale to better show the axes
    scale_x = np.linspace(min(coordX)*WL/2.0/np.pi, max(coordX)*WL/2.0/np.pi, npts)
    scale_z = np.linspace(min(coordZ)*WL/2.0/np.pi, max(coordZ)*WL/2.0/np.pi, npts)

    # Define scale ticks
    min_tick = min(min_tick, np.amin(Eabs_data))
    max_tick = max(max_tick, np.amax(Eabs_data))
    #max_tick = 5
    # scale_ticks = np.power(10.0, np.linspace(np.log10(min_tick), np.log10(max_tick), 6))
    scale_ticks = np.linspace(min_tick, max_tick, 11)

    # Interpolation can be 'nearest', 'bilinear' or 'bicubic'
    ax.set_title('Pabs')
    cax = ax.imshow(Eabs_data, interpolation = 'nearest', cmap = cm.jet,
                    origin = 'lower'
                    , vmin = min_tick, vmax = max_tick
                    , extent = (min(scale_x), max(scale_x), min(scale_z), max(scale_z))
                    #,norm = LogNorm()
                    )
    ax.axis("image")

    # Add colorbar
    cbar = fig.colorbar(cax, ticks = [a for a in scale_ticks])
    cbar.ax.set_yticklabels(['%5.3g' % (a) for a in scale_ticks]) # vertically oriented colorbar
    pos = list(cbar.ax.get_position().bounds)
    #fig.text(pos[0] - 0.02, 0.925, '|E|/|E$_0$|', fontsize = 14)
    if crossplane=='XZ':
        plt.xlabel('Z, nm')
        plt.ylabel('X, nm')
    elif crossplane=='YZ':
        plt.xlabel('Z, nm')
        plt.ylabel('Y, nm')
    

    # # This part draws the nanoshell
    from matplotlib import patches
    for xx in x:
        r= xx*WL/2.0/np.pi
        s1 = patches.Arc((0, 0), 2.0*r, 2.0*r,  angle=0.0, zorder=2,
                         theta1=0.0, theta2=360.0, linewidth=1, color='black')
        ax.add_patch(s1)

    from matplotlib.path import Path
    #import matplotlib.patches as patches

    # flow_total = 21
    # for flow in range(0,flow_total):
    #     flow_x, flow_z = GetFlow(scale_x, scale_z, Ec, Hc,
    #                              min(scale_x)+flow*(scale_x[-1]-scale_x[0])/(flow_total-1),
    #                                                 min(scale_z), npts, nmax=npts*10)
    #     verts = np.vstack((flow_z, flow_x)).transpose().tolist()
    #     #codes = [Path.CURVE4]*len(verts)
    #     codes = [Path.LINETO]*len(verts)
    #     codes[0] = Path.MOVETO
    #     path = Path(verts, codes)
    #     patch = patches.PathPatch(path, facecolor='none', lw=0.2, edgecolor='white',zorder = 2.5)
    #     ax.add_patch(patch)
    flow_total = 21
    scanSP = np.linspace(-factor*x[-1], factor*x[-1], npts)
    min_SP = -factor*x[-1]
    step_SP = 2.0*factor*x[-1]/(flow_total-1)
    x0, y0, z0 = 0, 0, 0
    max_length=factor*x[-1]*8
    max_angle = np.pi/200
    for flow in range(0,flow_total*2+1):
        if crossplane=='XZ':
            x0 = min_SP*2 + flow*step_SP
            z0 = min_SP
        elif crossplane=='YZ':
            y0 = min_SP*2 + flow*step_SP
            z0 = min_SP
        flow_xSP, flow_ySP, flow_zSP = GetFlow3D(x0, y0, z0, max_length, max_angle, x, m)
        if crossplane=='XZ':
            flow_z_plot = flow_zSP*WL/2.0/np.pi
            flow_x_plot = flow_xSP*WL/2.0/np.pi
        elif crossplane=='YZ':
            flow_z_plot = flow_zSP*WL/2.0/np.pi
            flow_x_plot = flow_ySP*WL/2.0/np.pi

        verts = np.vstack((flow_z_plot, flow_x_plot)).transpose().tolist()
        #codes = [Path.CURVE4]*len(verts)
        codes = [Path.LINETO]*len(verts)
        codes[0] = Path.MOVETO
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=0.2, edgecolor='white',zorder = 2.7)
        ax.add_patch(patch)
 
    plt.savefig("P-SiAgSi-flow-R"+str(int(round(x[-1]*WL/2.0/np.pi)))+"-"+crossplane+".pdf")
    plt.draw()

    plt.show()

    plt.clf()
    plt.close()
finally:
    terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(np.array([x]),
                                                                     np.array([m]))
    print("Qabs = "+str(Qabs));
#


