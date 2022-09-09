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
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file for virial')
    parser.add_argument('input2', metavar='INPUT', help='tst.dat file for pressure')
    #parser.add_argument('input3', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input4', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input5', metavar='INPUT', help='H5MD input file with data for state variables')

	
    args = parser.parse_args()

    H5 = [ h5py.File(args.input1, 'r'),  h5py.File(args.input2, 'r') ] #, h5py.File(args.input3, 'r'), h5py.File(args.input4, 'r'), h5py.File(args.input5, 'r')]
    pressure = []
    for j in range(len(H5)):
        H5obs = H5[j]['observables']
        pressure.append( [ H5obs['region{0}/pressure/value'.format(i)] for i in range(0,80) ] )

    box = np.diagonal(H5[0]['particles/all/box/edges']) 
    print(box) 
    slab = 0.4*box[0]
    Vtot = box[0]*box[1]*box[2]
    pore_area = [(15)**2, (25)**2]
    area = box[1]*box[2]
    dx =  2.5
    sym = -box[0]/2

    Vslab1 = int((box[0]-slab)//(2*dx))*[area*dx]
    Vslab1 += int(slab//dx)*[pore_area[0]*dx]
    Vslab1 += int((box[0]-slab)//(2*dx))*[area*dx]

    Vslab2 = int((box[0]-slab)//(2*dx))*[area*dx]
    Vslab2 += int(slab//dx)*[pore_area[1]*dx]
    Vslab2 += int((box[0]-slab)//(2*dx))*[area*dx]


    pressure =  np.array(pressure) #1x80x500
    #print(pressure.shape)
    mean_press = np.mean(pressure, axis = 2)#2 if more than 1 input

    p1 = (mean_press[0,:]*Vtot)/Vslab1
    p2 = (mean_press[1,:]*Vtot)/Vslab2
    #print(p1,p2)

    #import results from pressure_planes
    plane15 = np.loadtxt('tst15cst.dat')
    plane25 = np.loadtxt('tst25cst.dat')

    x15=plane15[:,0].tolist()
    x25=plane25[:,0].tolist()
    p15 = plane15[:,1].tolist()
    p25 = plane25[:,1].tolist()

    del x15[55], x25[55], p15[55], p25[55]

    

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
     
    #xgrid = np.loadtxt(args.input2)[:,0]
    xgrid = dx*np.arange(box[0]/dx)+1.25
    #print(xgrid)
        
    rdf0 = interp1d(xgrid, p1 ,bounds_error=False, kind = 'quadratic')
    rdf1 = interp1d(xgrid, p2 ,bounds_error=False, kind = 'quadratic')
    #rdf2 = interp1d(xgrid, mean_temp[2,:] ,bounds_error=False, kind = 'quadratic')
    #rdf3 = interp1d(xgrid, mean_temp[3,:] ,bounds_error=False, kind = 'quadratic')
    #rdf4 = interp1d(xgrid, mean_temp[4,:] ,bounds_error=False, kind = 'quadratic')

    
    grids_adr = np.linspace(0, box[0], num =1000 , endpoint=False)

    plt.plot(grids_adr + sym, rdf0(grids_adr) , '-',color='deepskyblue',linewidth=1.2,fillstyle='full', label = 'D15 virial')
    plt.plot(grids_adr + sym, rdf1(grids_adr) , '-',color='cornflowerblue',linewidth=1.2,fillstyle='full', label = 'D25 virial')
    plt.plot(x15, p15 , '-',color='red',linewidth=0.6 , linestyle='dashed', label = 'Planes method')
    plt.plot(x15, p25 , '-',color='red',linewidth=0.6,linestyle='dashed')
    #plt.plot(grids_adr + sym, rdf4(grids_adr) , '-',color='black',linewidth=1.2,fillstyle='full', label = 'L35')
    plt.legend()

    plt.xlabel(r"$x / \sigma$")
    plt.ylabel(r"$p/\varepsilon/\sigma^3$") 
    #print(np.max(time0)-0.8, np.max(time1)-1,np.max(time2)-1.2)
    source = 10
    slab = 0.4*box[0]
    plt.xlim([0+sym,box[0]+sym])
    #plt.ylim([0,5])
    plt.legend(loc = 'upper right')
    plt.axvline(x=source+sym, color='k', linestyle='--',linewidth=0.4)
    plt.axvspan(-slab/2, slab/2, alpha=0.5, color='grey')
    plt.axvspan(0+sym, source+sym, alpha=0.5, color='gold')

    plt.savefig('comp1525_nocorrection.pdf')
   



if __name__ == '__main__':
    main()
