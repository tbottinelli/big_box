#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright © 2011  Felix Höfling
#
# This file is part of HALMD.
#
# HALMD is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
#
from __future__ import division
import argparse
import h5py
import os
from numpy import *
#from pylab import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import math
from pynverse import inversefunc

def P_tail_correction(rho,rc,sigma):
    res = 32 * math.pi / 9 * rho*rho* ((sigma/rc)**9 - 1.5*(sigma/rc)**3)
    return res

def P_calculator_new_fixedT(rho):
    T = 1.06923206188 
    kB = 1; m = 1; Sigma = 1; epsilon = 1
    rho_reduced = rho*Sigma**3/m
    T_reduced = T*kB/epsilon
    landa = 3
    
    x = [0.8623085097507421, 2.976218765822098, -8.402230115796038, 0.1054136629203555, -0.8564583828174598,\
        1.582759470107601, 0.7639421948305453, 1.753173414312048, 2.798291772190376e3, -4.8394220260857657e-2,\
        0.9963265197721935, -3.698000291272493e1, 2.084012299434647e1, 8.305402124717285e1, -9.574799715203068e2,\
        -1.477746229234994e2, 6.398607852471505e1, 1.603993673294834e1, 6.805916615864377e1, -2.791293578795945e3,\
        -6.245128304568454, -8.116836104958410e3, 1.488735559561229e1, -1.059346754655084e4, -1.131607632802822e2,\
        -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2, -2.694523589434903e3,\
        -7.218487631550215e2, 1.721802063863269e2]

    a = [ x[0]*T_reduced + x[1]*math.sqrt(T_reduced) + x[2] + x[3]/T_reduced + x[4]/T_reduced**2,\
        x[5]*T_reduced + x[6] + x[7]/T_reduced + x[8]/T_reduced**2,\
        x[9]*T_reduced + x[10] + x[11]/T_reduced,\
        x[12],\
        x[13]/T_reduced + x[14]/T_reduced**2,\
        x[15]/T_reduced,\
        x[16]/T_reduced + x[17]/T_reduced**2,\
        x[18]/T_reduced**2]

    b = [ x[19]/T_reduced**2 + x[20]/T_reduced**3,\
        x[21]/T_reduced**2 + x[22]/T_reduced**4,\
        x[23]/T_reduced**2 + x[24]/T_reduced**3,\
        x[25]/T_reduced**2 + x[26]/T_reduced**4,\
        x[27]/T_reduced**2 + x[28]/T_reduced**3,\
        x[29]/T_reduced**2 + x[30]/T_reduced**3 + x[31]/T_reduced**4 ]
        
    first_sum = 0; second_sum = 0
    for i in range(8): first_sum += a[i]*rho_reduced**(i+2)
    for i in range(6): second_sum += b[i]*rho_reduced**(2*i+3)  
    
    P_reduced = rho_reduced*T_reduced + first_sum + math.exp(-landa*rho_reduced**2)*second_sum
    
    return P_reduced*epsilon/Sigma**3

def P_calculator_new(rho, T):
    kB = 1; m = 1; Sigma = 1; epsilon = 1
    rho_reduced = rho*Sigma**3/m
    T_reduced = T*kB/epsilon
    landa = 3
    
    x = [0.8623085097507421, 2.976218765822098, -8.402230115796038, 0.1054136629203555, -0.8564583828174598,\
        1.582759470107601, 0.7639421948305453, 1.753173414312048, 2.798291772190376e3, -4.8394220260857657e-2,\
        0.9963265197721935, -3.698000291272493e1, 2.084012299434647e1, 8.305402124717285e1, -9.574799715203068e2,\
        -1.477746229234994e2, 6.398607852471505e1, 1.603993673294834e1, 6.805916615864377e1, -2.791293578795945e3,\
        -6.245128304568454, -8.116836104958410e3, 1.488735559561229e1, -1.059346754655084e4, -1.131607632802822e2,\
        -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2, -2.694523589434903e3,\
        -7.218487631550215e2, 1.721802063863269e2]

    a = [ x[0]*T_reduced + x[1]*math.sqrt(T_reduced) + x[2] + x[3]/T_reduced + x[4]/T_reduced**2,\
        x[5]*T_reduced + x[6] + x[7]/T_reduced + x[8]/T_reduced**2,\
        x[9]*T_reduced + x[10] + x[11]/T_reduced,\
        x[12],\
        x[13]/T_reduced + x[14]/T_reduced**2,\
        x[15]/T_reduced,\
        x[16]/T_reduced + x[17]/T_reduced**2,\
        x[18]/T_reduced**2]

    b = [ x[19]/T_reduced**2 + x[20]/T_reduced**3,\
        x[21]/T_reduced**2 + x[22]/T_reduced**4,\
        x[23]/T_reduced**2 + x[24]/T_reduced**3,\
        x[25]/T_reduced**2 + x[26]/T_reduced**4,\
        x[27]/T_reduced**2 + x[28]/T_reduced**3,\
        x[29]/T_reduced**2 + x[30]/T_reduced**3 + x[31]/T_reduced**4 ]
        
    first_sum = 0; second_sum = 0
    for i in range(8): first_sum += a[i]*rho_reduced**(i+2)
    for i in range(6): second_sum += b[i]*rho_reduced**(2*i+3)  
    
    P_reduced = rho_reduced*T_reduced + first_sum + math.exp(-landa*rho_reduced**2)*second_sum
    
    return P_reduced*epsilon/Sigma**3

def grad_P_calculator_new(rho, T):
    kB = 1; m = 1; Sigma = 1; epsilon = 1
    rho_reduced = rho*Sigma**3/m
    T_reduced = T*kB/epsilon
    landa = 3
    
    x = [0.8623085097507421, 2.976218765822098, -8.402230115796038, 0.1054136629203555, -0.8564583828174598,\
        1.582759470107601, 0.7639421948305453, 1.753173414312048, 2.798291772190376e3, -4.8394220260857657e-2,\
        0.9963265197721935, -3.698000291272493e1, 2.084012299434647e1, 8.305402124717285e1, -9.574799715203068e2,\
        -1.477746229234994e2, 6.398607852471505e1, 1.603993673294834e1, 6.805916615864377e1, -2.791293578795945e3,\
        -6.245128304568454, -8.116836104958410e3, 1.488735559561229e1, -1.059346754655084e4, -1.131607632802822e2,\
        -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2, -2.694523589434903e3,\
        -7.218487631550215e2, 1.721802063863269e2]

    a = [ x[0]*T_reduced + x[1]*math.sqrt(T_reduced) + x[2] + x[3]/T_reduced + x[4]/T_reduced**2,\
        x[5]*T_reduced + x[6] + x[7]/T_reduced + x[8]/T_reduced**2,\
        x[9]*T_reduced + x[10] + x[11]/T_reduced,\
        x[12],\
        x[13]/T_reduced + x[14]/T_reduced**2,\
        x[15]/T_reduced,\
        x[16]/T_reduced + x[17]/T_reduced**2,\
        x[18]/T_reduced**2]

    b = [ x[19]/T_reduced**2 + x[20]/T_reduced**3,\
        x[21]/T_reduced**2 + x[22]/T_reduced**4,\
        x[23]/T_reduced**2 + x[24]/T_reduced**3,\
        x[25]/T_reduced**2 + x[26]/T_reduced**4,\
        x[27]/T_reduced**2 + x[28]/T_reduced**3,\
        x[29]/T_reduced**2 + x[30]/T_reduced**3 + x[31]/T_reduced**4 ]
        
    first_sum = 0; second_sum = 0; third_sum = 0
    for i in range(8): first_sum += a[i]*(i+2)*rho_reduced**(i+1)
    for i in range(6): second_sum += b[i]*rho_reduced**(2*i+3)
    for i in range(6): third_sum += b[i]*(2*i+3)*rho_reduced**(2*i+2)

    P_reduced = T_reduced + first_sum + math.exp(-landa*rho_reduced**2)*third_sum - 2*landa*rho_reduced*math.exp(-landa*rho_reduced**2)*second_sum
    
    return P_reduced*epsilon/Sigma**3

'''not used here

def compute_density_profile(wavevector_list, density_modes_list, box, width, user_axis):
    one_norm=np.linalg.norm(wavevector_list,axis=1,ord=1)
    wavevector_axis_component=wavevector_list[:,user_axis]
    idx, = np.where(one_norm == np.abs(wavevector_axis_component))

    wavevector_list_of_interest=wavevector_axis_component[idx]
    density_modes_list_of_interest=density_modes_list[idx]

    # generating and applying 1D-Gaussian filter, width = 0 disables the filter (gaussian = 1)
    gaussian = np.exp(-0.5 * width**2 * pow(wavevector_list_of_interest, 2))
    density_modes_smoothed=density_modes_list_of_interest * gaussian

    # In general (arbitrary dimensions) the density modes are a scalar field on the space of the wavevectors k.
    # The wavevectors forming a cubic grid around 0 (with halmd.observables.utility.wavevector(...,dense=true)) 

    # carry the information of the position in k-space of each density mode.
    # This information can be used to restructure the list of density modes to a density mode matrix. 
    # Hereby transforming the wavevectors by some factor to the set of smallest integers w, 
    # produces the index to locate the density-modes in a matrix.
    w = np.array(np.round(wavevector_list_of_interest * box[user_axis] / (2 * np.pi)), dtype=int)

    # initialising and filling the 1D-density_modes_matrix
    assert np.min(w)==-np.max(w) , "Density-modes need to be on a symmetric grid around 0, i.e. k_max = -k_min"
    # print w to see whether it covers the whole box 
    length = 2*np.max(w)+1
    density_modes_matrix = np.zeros(length, dtype=complex)
    density_modes_matrix[w] = density_modes_smoothed

    # Fourier backtransform
    density_unnormalized = np.fft.fftshift(np.fft.ifft(density_modes_matrix)).real

    # normalisation
    # even though a one dimensional fft was done, the density_mode_values on the desired coordinate-axis
    # represent the density averaged over the other coordinate axes. Therefore after the inverse Fourier transform, they correspond 
    # to an integral of the density field in the other coordinate axes by one number, but still refer to the full-dimensional box. 
    volume = np.prod(box)  
    density = density_unnormalized * len(density_unnormalized) / volume

    #Generate the corresponding positions in real space, from the wavevectors
    position = box[user_axis] / np.double(length)  * np.linspace(np.min(w), np.max(w), length) 

    return position, density
'''


def main():
    # define and parse command line arguments
    parser = argparse.ArgumentParser(prog='temp_plotter.py')
    parser.add_argument('--range', type=int, nargs=2, help='select range of data points')
    parser.add_argument('--dump', metavar='FILENAME', help='dump plot data to filename')
    parser.add_argument('--no-plot', action='store_true', help='do not produce plots, but do the analysis')
    parser.add_argument('--group', help='particle group (default: %(default)s)', default='all')
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file with data for state variables')#
#	parser.add_argument('input2', metavar='INPUT', help='H5MD input file with data for state variables')
	
    args = parser.parse_args()

    plt.rc('font', **{ 'family':'serif', 'serif' : ['ptm'], 'size' :12})
    plt.rc('text', usetex=True)
    plt.rc('text.latex' , preamble=(
        r'\usepackage{textcomp}',
        r'\usepackage{amsmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{times}',
        r'\usepackage{txfonts}',
        ))
    plt.rc('legend', frameon=False, numpoints=1, fontsize=8, labelspacing=0.2, handlelength=2, handletextpad=0.5, borderaxespad=0.5)
    plt.rc('figure',figsize=(4.7,2))
    plt.rc('xtick', direction='in',top=True)
    plt.rc('ytick', direction='in',right=True)
    plt.rc('xtick.minor',visible=True,top=True)
    plt.rc('ytick.minor',visible=True,right=True)
    plt.rc('axes', linewidth=0.7 )
    plt.rc('lines', linewidth=1, markersize = 2,markeredgewidth=0)
    plt.rc('savefig', bbox='tight',pad_inches=0.05,dpi=600,transparent=False)
    plt.rc('ps',usedistiller='xpdf')


    H5 = h5py.File(args.input1)
    box = np.diagonal(H5['particles/all/box/edges'])
    sym = -box/2
    area = box[1]*box[2]
    slab = 0.4*box[0]
    pore = 25
    pore_area = (pore)**2
    dx = 2.5
    xgrid = np.arange(sym[0], box[0] + sym[0], dx) 
    print(xgrid.shape) #80
    
    nknots_adr = [801, 2, 2]   

    dx_adr = box[0] / (nknots_adr[0] - 1) #0.25
    xgrid_adr = np.arange(sym[0], box[0]+sym[0], dx_adr) 
    print(xgrid_adr.shape)

    Vtot = box[0]*box[1]*box[2]
    Vslab = int((box[0]-slab)//(2*dx))*[area*dx]
    Vslab += int(slab//dx)*[pore_area*dx]
    Vslab += int((box[0]-slab)//(2*dx))*[area*dx]


#################calculation of pressure
    H5 = h5py.File(args.input1)

    H5obs = H5['observables']

    press = []
    press.append( np.mean([ H5obs['region{0}/pressure/value'.format(i)] for i in range(0,80)], axis = 1) )
    p = ( np.array(press)*Vtot)/Vslab
    p = p[0,:]
    print('press', p.shape)

###########calculation of modified temperature

    temp = []
    temp.append( np.mean([ H5obs['region{0}/temperature/value'.format(i)] for i in range(0,80)], axis = 1) )
    temp = np.array(temp)
    #temp = temp[0,:]
    print('temp', temp.shape)

    rdfTemp = interp1d(xgrid, temp[0,:], bounds_error=False, kind = 'quadratic')

    grids_adr = np.linspace(0, box[0], num =1000 , endpoint=False)

    #plt.plot(xgrid_adr, rdfTemp(xgrid_adr) , '-', color="blueviolet",linewidth=1.2, label='temp')
##########calculation of density

    nknots_adr = [801, 2, 2]   

    dx_adr = box[0] / (nknots_adr[0] - 1) #0.25
    xgrid_adr = np.arange(sym[0], box[0]+sym[0], dx_adr) 
    print(xgrid_adr, xgrid_adr.shape)

    p_num = []
    p_num.append( np.mean([ H5obs['region{0}/particle_number/value'.format(i)] for i in range(0,80)], axis = 1) )
    p_num = np.array(p_num)
    rho = p_num / Vslab
    rho = rho[0,:]
    print('density', rho.shape)

    rdfRho = interp1d(xgrid, rho, bounds_error=False, kind = 'quadratic')

    rho = np.array([rdfRho(xgrid_adr[:])])
    print(rho, rho.shape)
    
    plt.plot(xgrid_adr, rdfRho(xgrid_adr) , '-', color="royalblue",linewidth=1.2, label=r'$\rho$')

##########EQS

    rc = 2.5; sigma = 1

    res_rho = []
    res_rho_c = []
    grid = []
    for idx in range(0,80):
        x = 0.78
        x_c = 0.78
        p0 = p[idx]
        p0_c = p[idx] + P_tail_correction(rho[0,int((idx*2.5+1.25)//0.25)],rc,sigma)
        T0 = rdfTemp(xgrid[idx])
        for i in range(0,1000):
            x_new = x - (P_calculator_new(x,T0) - p0)/grad_P_calculator_new(x,T0)
            x_new_c = x_c - (P_calculator_new(x_c,T0) - p0_c)/grad_P_calculator_new(x_c,T0)
            #print("step", i , ":" ,x_new)
            x = x_new
            x_c = x_new_c
        #resulted_rho[int((idx*2.5+1.25)/0.25)]=x
        res_rho.append(x)
        res_rho_c.append(x_c)
        grid.append(xgrid_adr[int((idx*2.5+1.25)//0.25)])
        print("idx:",idx)
    #plt.plot(grid, res_rho,'o',color='b',linewidth=1.2,label=r"$\rho_{EQS}(T,P)$")
    plt.plot(grid, res_rho_c,"o",color='r',linewidth=1.2,label=r"$\rho_{EQS}(T,P+P_{c})$")
    #print(res_rho,grid)

    np.savetxt("sigma2.2.txt",res_rho_c,delimiter = ",")
    np.savetxt("grid.txt",grid,delimiter = ",")

    #plt.legend(loc='upper left',prop={'size': 6})
    plt.xlabel(r"$x / \sigma$")
    plt.ylabel(r"$\rho$") 
    #plt.ylabel(r"$k_{B}T(x)/\varepsilon$") 
    #print(time0)

    source = 10
    plt.xlim([0+sym[0],box[0]+sym[0]])
    #plt.ylim([0,3])
    plt.legend(loc = 'lower left')
    plt.axvline(x=source+sym[0], color='k', linestyle='--',linewidth=0.4)
    
    plt.axvspan(-slab/2, slab/2, alpha=0.5, color='grey')
    #plt.axvspan(0+sym, TR+Delta+sym, alpha=0.5, color='tomato')
	
    '''plt.axvline(x=35.0, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=37.5, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=57.5, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=60.0, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=95.0, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=97.5, color='k', linestyle=':', linewidth=0.7)
    plt.axvline(x=117.5, color='k', linestyle=':', linewidth=0.7)'''
    
    plt.axvspan(sym[0],source+sym[0], alpha=0.5, color='gold')
    #plt.axvspan(source+thermo+hm+sym,source+2*thermo+hm+sym, alpha=0.5, color='tomato')
    #plt.axvspan(57.5, 60, alpha=0.5, color='skyblue')

   
    #fig.canvas.draw()
#    plt.savefig('mc_vs_no_mc_momntm_den_noneq_rho0.6_T1_1.5.pdf')
    plt.savefig('p_25.pdf')
   


if __name__ == '__main__':
    main()
