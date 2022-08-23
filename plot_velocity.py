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
    parser = argparse.ArgumentParser(prog='plot_velocity.py')
    parser.add_argument('--range', type=int, nargs=2, help='select range of data points')
    parser.add_argument('--dump', metavar='FILENAME', help='dump plot data to filename')
    parser.add_argument('--no-plot', action='store_true', help='do not produce plots, but do the analysis')
    parser.add_argument('--group', help='particle group (default: %(default)s)', default='all')
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file with data for state variables')
    parser.add_argument('input2', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input3', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input4', metavar='INPUT', help='H5MD input file with data for state variables')

    args = parser.parse_args()
    H5 = [ h5py.File(args.input1, 'r'), h5py.File(args.input2, 'r') ]

    velocity = []
    position = []
    images = []
    species = []
    for i in range(len(H5)):
        H5p = H5[i]['particles/all']
        velocity.append([ H5p['velocity/value'] ]) 
        velocity = np.array(velocity)
        position.append(H5p['position/value']) 
        position = np.array(position)
        images.append(H5p['image/value'])
        images = np.array(images)
        species.append(H5p['species/value'])
        species = np.array(species)
        print('pos',position.shape)
        print('vel',velocity.shape)
        print('species',species.shape)
    
    box = np.diagonal(H5[0]['particles/all/box/edges'])
    slab = 0.4*box[0]
    slab_start = -slab/2
    print(slab_start)
    slab_end = slab/2
    pore_yz = 15
    print(pore_yz)
    sym = box/(-2)

    #bring particles back in the box
    for i in range(3):
        position[:,:,i] -= box[i]*images[:,:,i] 
    print('pos',position.shape[0])

    fluid_id = np.where((species[0,:] == 0))
    print('fluid id',fluid_id[0].shape)

    fluid_pos = []
    speed = []
    for t in range(position.shape[0]):
        #selecting fluid particles in the length of the pore (then select in the y direction)
        fluid_pos.append( np.where( abs(position[t, fluid_id[0], 0] ) < slab/2, position[t, fluid_id[0], 1] , np.nan ) )
        #speed in x direction from these particles
        speed.append( np.where( abs(position[t, fluid_id[0], 0] ) < slab/2, velocity[t, fluid_id[0], 0], np.nan ))
        
    fluid_pos = np.nanmean( np.array(fluid_pos), axis = 0 )
    speed = np.nanmean( np.array(speed), axis = 0 ) 

    fluid_pos = fluid_pos[ np.where(np.isnan(fluid_pos) == False)]
    speed = speed[ np.where(np.isnan(speed) == False)]
    print('fluid in pore position y', fluid_pos.shape)
    print('speed x', speed.shape)

    #create slabs in y direction to average velocity
    dy =  np.arange(np.amin(fluid_pos), np.amax(fluid_pos)-0.5, 0.5)
    print(dy) 

    velocity = []
    for i in range(len(dy)) :
        pos_dy = np.array( np.where( (fluid_pos  >  dy[i]) & (fluid_pos < dy[i]+0.5) ))
        #print('pos_id',pos_y[pos_dy])
        #print('speed x',speed_x[pos_dy])
        velocity.append( np.mean(speed[pos_dy]) )
    print(velocity)
        

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
     
        
    plt.plot(dy, velocity, '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D15')
    #plt.plot(grids_adr + sym, rdf1(grids_adr) , '-',color='royalblue',linewidth=1.2,fillstyle='full', label = 'D10')
    #plt.plot(grids_adr + sym, rdf2(grids_adr) , '-',color='mediumblue',linewidth=1.2,fillstyle='full', label = 'D15')
    #plt.plot(grids_adr + sym, rdf3(grids_adr) , '-',color='midnightblue',linewidth=1.2,fillstyle='full', label = 'D20')
    plt.legend()

    plt.xlabel('y')
    plt.ylabel('velocity x') 
    #print(np.max(time0)-0.8, np.max(time1)-1,np.max(time2)-1.2)
    source = 10
    plt.xlim([0+sym[1],box[1]+sym[1]])
    #plt.ylim([-1,1])
    #plt.legend(loc = 'upper left')
    #plt.axvline(x=source+sym, color='k', linestyle='--',linewidth=0.4)
 

    plt.axvspan(-pore_yz/2,pore_yz/2, alpha=0.5, color='grey')
   # plt.axvspan(0+sym, source+sym, alpha=0.5, color='gold')

    plt.savefig('velocity.pdf')
   



if __name__ == '__main__':
    main()
