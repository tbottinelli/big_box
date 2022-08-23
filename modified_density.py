#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#copyright  © 2019 Abbas Gholami, Roya Ebrahimi
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

import numpy as np
import os
import h5py
import subprocess
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from pylab import rc, cycler
#from palette import LondonUnderground_Colours
import math

def compute_density_profile(wavevector_list, density_modes_list, box_edges, width, user_axis):

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
    w = np.array(np.round(wavevector_list_of_interest * box_edges[user_axis] / (2 * np.pi)), dtype=int)

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
    volume = np.prod(box_edges)  
    density = density_unnormalized * len(density_unnormalized) / volume

    #Generate the corresponding positions in real space, from the wavevectors
    position = box_edges[user_axis] / np.double(length)  * np.linspace(np.min(w), np.max(w), length) 

    return position, density

def full(n_circle, radius, position):
    output = n_circle * math.pi * (radius*radius - position*position)
    return output

def main():

    import shutil
    import scipy.linalg as la
    
    parser = argparse.ArgumentParser(prog='density_plotter.py')
    parser.add_argument('input', metavar='INPUT', help='H5MD input file with data for state variables')
    args = parser.parse_args()

    source = 10

    #parameters
    nknots = [801, 2, 2]
    C = 1#thermodynamic force correction rate (between 0 and 1)
    density_accuracy = 0.5 #the maximum accuracy for density profile to reach at each step
    
    coefficients = np.zeros(8*np.prod(nknots))
    #opening H5MD file for reading box and particle number data
    
    with h5py.File(args.input, 'r') as h5:
        box_edges = np.diagonal(h5["particles/all/box/edges"])

    dx = box_edges[0] / (nknots[0] - 1)
    plt.rc('font', **{ 'family':'serif', 'serif' : ['ptm'], 'size' :8.5})
    plt.rc('text', usetex=True)
    plt.rc('text.latex' , preamble=(
        r'\usepackage{textcomp}',
        r'\usepackage{amsmath}',
        r'\usepackage[T1]{fontenc}',
        r'\usepackage{times}',
        r'\usepackage{txfonts}',
        ))
    plt.rc('legend', frameon= False, numpoints=1, fontsize=8, labelspacing=0.2, handlelength=2, handletextpad=0.5, borderaxespad=0.5)
    plt.rc('figure',figsize=(4.7,2.0))
    plt.rc('xtick', direction='in',top=True)
    plt.rc('ytick', direction='in',right=True)
    plt.rc('xtick.minor',top=True,visible=True)
    plt.rc('ytick.minor',right=True,visible=True)
    plt.rc('axes', linewidth=0.7 )
    plt.rc('lines', linewidth=1, markersize = 2,markeredgewidth=0)
    plt.rc('savefig', bbox='tight',pad_inches=0.05,dpi=600,transparent=False)
    plt.rc('ps',usedistiller='xpdf')
    plt.xlabel(r"$x / \sigma$")
    plt.ylabel(r"$\rho(x) \,\sigma^3$")
    #plt.title(r"$T=1,j=0.1$")
  
    density_sum = 0
    b=0
    s=0
    a=0
    e=0
    sp = -3.75
    ep = 13.75
    lattice_size = 1.25
    N_circle = 72
    A = 30 * 30
    indexes = []
    #read density modes from .h5 file

    sym=-50
    xgrid = dx * np.arange(int((100)/dx)+1)
    xgrid -= 50
    for i in range(1,2):    
        h5 = h5py.File('result_15D15.h5', 'r') 
        wavevector = np.array(h5['structure/allfluid/density_mode/wavevector'])
        density_modes = np.mean(h5['structure/allfluid/density_mode/value'], axis=0)
        print(density_modes)
        density_modes = density_modes[..., 0] + 1j * density_modes[..., 1]
        #compute density profile
        mesh, density = compute_density_profile(wavevector, density_modes, box_edges, width = 0.35, user_axis = 0)
        density_sum += density
        b+=1
        print (b)
        print(density_sum)


    density_sum = [j/(b * 1.0) for j in density_sum]
   # print(density_sum)
    #epsilon = 0.05
    r = 1.384 #+ epsilon
    for j in range (len(xgrid)):
        if ep + r >= xgrid[j] >= sp - r :
            indexes.append(j)

    l = lattice_size/2
    for j in indexes:
        x = xgrid[j] - sp
        x -= int(x/l) * l
        if x <= r :
            f = (A-full(N_circle,r,x))/A
            density_sum[j]=density_sum[j]/f
            print(f)
        elif l-r <= x:
            f = (A-full(N_circle,r,x-l))/A
            density_sum[j]=density_sum[j]/f
            print(f)
        elif x>=l:
            print("there is a mistake")
    print(density_sum)
    plt.plot(xgrid, density_sum,'-',color='teal',linewidth=1,fillstyle = 'none',label= r"$J = 0.05$")

    #plotting
    dirName = 'result'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    plt.ion()

    #yticks = np.arange(0.68,0.84,0.03)
    #plt.yticks(yticks)
    #plt.ylim([0.45,1.25])
    plt.xlim([0+sym,100+sym])
    #plt.xlim([-10,20])

    plt.axvline(x=source+sym,  color='k', linestyle='--',linewidth=0.4)
    plt.axvspan(-3.75, 13.75, alpha=0.5, color='grey')
    plt.axvspan(0+sym,source+sym, alpha=0.5, color='gold')
    plt.show()
    plt.savefig("density_newpar.pdf")




if __name__ == '__main__':
    main()
