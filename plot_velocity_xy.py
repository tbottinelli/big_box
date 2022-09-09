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
from scipy.optimize import curve_fit

def NS(dy,visc):
    deltax= 80
    dp_25 = 2.449 - 4.097
    dp_15 = 11.0995 - 15.927
    dpdx = [dp_15/deltax, dp_25/deltax]
    R = [15/2, 25/2]

    return (-(dpdx[0]*R[0]**2)/(4*visc))*(1-(dy/R[0])**2)


def main():
    # define and parse command line arguments
    parser = argparse.ArgumentParser(prog='plot_velocity_xy.py')
    parser.add_argument('--range', type=int, nargs=2, help='select range of data points')
    parser.add_argument('--dump', metavar='FILENAME', help='dump plot data to filename')
    parser.add_argument('--no-plot', action='store_true', help='do not produce plots, but do the analysis')
    parser.add_argument('--group', help='particle group (default: %(default)s)', default='all')
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input2', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input3', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input4', metavar='INPUT', help='H5MD input file with data for state variables')

    args = parser.parse_args()
    H5 = h5py.File('result_out_20220908_104237')

    H5p = H5['particles/all']
    velocity =  H5p['velocity/value']
    position = H5p['position/value']
    images = H5p['image/value']
    species = H5p['species/value']
    print('pos',position.shape)
    print('vel',velocity.shape)
    print('species',species.shape)
    print('images', images.shape)

    box = np.array(np.diagonal(H5['particles/all/box/edges']))
    print(box)
    slab = 0.4*box[0]
    print(slab/2)
    slab_start = -slab/2
    slab_end = slab/2
    pore_yz = 15
    print(pore_yz)
    sym = -box/2


    num_times = position.shape[0]

    #bring particles back in the box
    for i in range(3):
        position[:,:,i] -= box[i]*images[:,:,i] 
    print(np.amax(position), np.amin(position))

    fluid_id = np.where((species[0,:] == 0))
    print('fluid id',fluid_id[0].shape)
    
    #print(abs(position[0,fluid_id[0], 0]))


    fluid_posy = []
    speed = []
    for t in range(num_times):
    #selecting fluid particles in the length of the pore (then select in the y direction)
        fluid_posy.append( np.where( abs(position[t, fluid_id[0], 0] ) < slab/2, position[t, fluid_id[0], 1] , np.nan ) )
        #print(np.nanmax(fluid_posy), np.nanmin(fluid_posy))
        #speed in x direction from these particles
        speed.append( np.where( abs(position[t, fluid_id[0], 0] ) < slab/2, velocity[t, fluid_id[0], 0], np.nan )) 
    fluid_posy = np.nanmean( np.array(fluid_posy), axis = 0 )
    speed = np.nanmean( np.array(speed), axis = 0 ) 

    fluid_posy = fluid_posy[ np.where(np.isnan(fluid_posy) == False)]
    speed = speed[ np.where(np.isnan(speed) == False)]
    print('posx pos', fluid_posy)
    print('speed ', speed)

    #create slabs in y direction to average velocity
    dy =  np.arange(np.amin(fluid_posy) + 0.5, np.amax(fluid_posy), 0.5 ) 
    print('dy', dy) 
    
    velocity = []
    for x in range(len(dy)) :
        pos_dy = np.array( np.where( (fluid_posy  >  dy[x]-0.5) & (fluid_posy < dy[x]) ))
        #print(pos_dx)
        #print(fluid_posy[pos_dx])
        velocity.append( np.mean(speed[pos_dy] ))
    print(velocity)

    #visc15, cov15 = curve_fit(NS, data15[:,0], data15[:,1])
    #visc25, cov25 = curve_fit(NS, data25[:,0], data25[:,1])


    plt.rc('font', **{ 'family':'serif', 'serif' : ['ptm'], 'size' :12})
    species = np.array(species)
    species = np.array(species)
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
     
    dy = np.array(data15[:,0])-0.25
    plt.plot(dy, data15[:,1], '-',color='deepskyblue', linewidth=1.2,fillstyle='full', label = 'D=15')
    plt.plot(dy-0.25, NS(dy, visc15) ,  '-',color='red',linewidth=0.8,linestyle='dashed', label = 'NS fit viscosity = %5.3f' %visc15)
    #plt.plot( dy, velocity_mean[2,:], '-',color='mediumblue',linewidth=1.2,fillstyle='full', label = 'x=90')
    #plt.plot(dy, velocity_mean[3,:] , '-',color='midnightblue',linewidth=1.2,fillstyle='full', label = 'x=110')
    plt.legend()

    plt.xlabel('x')
    plt.ylabel('velocity x') 
    #print(np.max(time0)-0.8, np.max(time1)-1,np.max(time2)-1.2)
    source = 10
    plt.xlim([0+sym[1],box[1]+sym[1]])
    #plt.ylim([-1,1])
    #plt.legend(loc = 'upper left')
    #plt.axvline(x=source+sym, color='k', linestyle='--',linewidth=0.4)
 

    plt.axvspan(-pore_yz/2,pore_yz/2, alpha=0.5, color='grey')

   # plt.axvspan(0+sym, source+sym, alpha=0.5, color='gold')

    plt.savefig('veltot15.pdf')
   



if __name__ == '__main__':
    main()
