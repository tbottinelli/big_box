#!/usr/bin/env python
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

def main():
    # define and parse command line arguments
    parser = argparse.ArgumentParser(prog='plot_temp.py')
    parser.add_argument('--range', type=int, nargs=2, help='select range of data points')
    parser.add_argument('--dump', metavar='FILENAME', help='dump plot data to filename')
    parser.add_argument('--no-plot', action='store_true', help='do not produce plots, but do the analysis')
    parser.add_argument('--group', help='particle group (default: %(default)s)', default='all')
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input2', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input3', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input4', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input5', metavar='INPUT', help='H5MD input file with data for state variables')

	
    args = parser.parse_args()
    eta = 1.4


    H5 =  h5py.File(args.input1, 'r') #,  h5py.File(args.input2, 'r') ] #, h5py.File(args.input3, 'r'), h5py.File(args.input4, 'r'), h5py.File(args.input5, 'r')]
    box = np.diagonal(H5['particles/all/box/edges'])
    sym = - box/2
    H5obs = H5['observables']
    Temp = np.array([  H5obs['region{0}/temperature/value'.format(i)] for i in range(0,80) ] )
    J = np.array([ H5obs['region{0}/heat_flux/value'.format(i)] for i in range(0,80) ])
    kin = np.array([ H5obs['region{0}/kinetic_energy/value'.format(i)] for i in range(0,80) ])
    Uint = np.array([ H5obs['region{0}/internal_energy/value'.format(i)] for i in range(0,80) ])
    stress = np.array([H5obs['region{0}/stress_tensor/value'.format(i)] for i in range(0,80) ]) #need component 3 and 4 [:,:,3] []
    velocity = np.array([H5obs['region{0}/center_of_mass_velocity/value'.format(i)] for i in range(0,80) ])
    
    

    summation = Uint + J[:,:,0]*Temp + stress .....
    print(summation.shape)
    sum_mean = np.mean(summation, axis = 1)

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
    
    dx=2.5
    xgrid = dx*np.arange(int(box[0]/dx))+1.25
        
    rdf0 = interp1d(xgrid, sum_mean ,bounds_error=False, kind = 'quadratic')
 #   rdf1 = interp1d(xgrid, mean_temp[1,:] ,bounds_error=False, kind = 'quadratic')
    #rdf2 = interp1d(xgrid, mean_temp[2,:] ,bounds_error=False, kind = 'quadratic')
    #rdf3 = interp1d(xgrid, mean_temp[3,:] ,bounds_error=False, kind = 'quadratic')
    #rdf4 = interp1d(xgrid, mean_temp[4,:] ,bounds_error=False, kind = 'quadratic')

    
    grids_adr = np.linspace(0,box[0], num =1000 , endpoint=False)
    sym = -box/2

    plt.plot(grids_adr + sym[0], rdf0(grids_adr) , '-',color='deepskyblue',linewidth=1.2,fillstyle='full', label = 'D15')
#    plt.plot(grids_adr + sym[0], rdf1(grids_adr) , '-',color='royalblue',linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(grids_adr + sym, rdf2(grids_adr) , '-',color='mediumblue',linewidth=1.2,fillstyle='full', label = 'L25')
    #plt.plot(grids_adr + sym, rdf3(grids_adr) , '-',color='midnightblue',linewidth=1.2,fillstyle='full', label = 'L30')
    #plt.plot(grids_adr + sym, rdf4(grids_adr) , '-',color='black',linewidth=1.2,fillstyle='full', label = 'L35')
    plt.legend()

    plt.xlabel(r"$x / \sigma$")
    plt.ylabel(r"$k_{B}T(x)/\varepsilon$") 
    #print(np.max(time0)-0.8, np.max(time1)-1,np.max(time2)-1.2)
    source = 10
    slab = 0.4*box[0]
    plt.xlim([0+sym[0],box[0]+sym[0]])
    #plt.ylim([0,5])
    #plt.legend(loc = 'upper left')
    plt.axvline(x=source+sym[0], color='k', linestyle='--',linewidth=0.4)
 

    plt.axvspan(-slab/2,slab/2, alpha=0.5, color='grey')
    plt.axvspan(0+sym[0], source+sym[0], alpha=0.5, color='gold')

    plt.savefig('Esum.pdf')
   



if __name__ == '__main__':
    main()
