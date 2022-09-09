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
from scipy.optimize import curve_fit

#let's test if it's coherent with navier-stokes
def NS(dy, visc):

    deltax = 80
    dp_25= 2.449 - 4.097
    dp_15 = 11.0995 -  15.927
    dpdx =[ dp_15/deltax, dp_25/deltax]
    R = [15/2, 25/2]

    return (-(dpdx[0]*R[0]**2)/(4*visc))*(1-(dy/R[0])**2) 


def yaxis(pos, vel):
    
    step = 0.5
    #create slabs in y direction to average velocity
    dy = step * np.arange( (np.amax(pos) - np.amin(pos))//step) + step - np.amax(pos)
    #print(dy)
    velocity = []
    for i in range(len(dy)) :
        pos_dy = np.array( np.where( (pos  >  dy[i]-step) & (pos < dy[i]) ))
        #print('pos_id', pos_dy.shape)
        velocity.append( np.mean(vel[pos_dy]) )
    #print(velocity, len(velocity))
    return velocity, dy
    

def main():
    # define and parse command line arguments
    parser = argparse.ArgumentParser(prog='plot_velocity.py')
    parser.add_argument('--range', type=int, nargs=2, help='select range of data points')
    parser.add_argument('--dump', metavar='FILENAME', help='dump plot data to filename')
    parser.add_argument('--no-plot', action='store_true', help='do not produce plots, but do the analysis')
    parser.add_argument('--group', help='particle group (default: %(default)s)', default='all')
    parser.add_argument('input1', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input2', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input3', metavar='INPUT', help='H5MD input file with data for state variables')
    #parser.add_argument('input4', metavar='INPUT', help='H5MD input file with data for state variables')
    args = parser.parse_args()

    every = 50
    H5 = h5py.File(args.input1, 'r')
    H5vel =  H5['particles/all/velocity']
    velocity = np.array(H5vel['value'])[::every, :,:]
    H5pos = H5['particles/all/position']
    position = np.array(H5pos['value'])[::every, :,:]
    Himage = H5['particles/all/image']
    images = np.array(Himage['value'])[::every, :,:]
    H5species = H5['particles/all/species']
    print('pos',position.shape)
    species = np.array(H5species['value'])[::every,:]

    print('vel',velocity.shape)
    print('species',species.shape)
    
    box = np.diagonal(H5['particles/all/box/edges'])
    print(box)
    slab = 0.4*box[0]
    slab_start = -slab/2
    #print(slab_start)
    slab_end = slab/2
    pore_yz = 15
    print(pore_yz)
    sym = box/(-2)
    
    #bring particles back in the box
    for i in range(3):
        position[:,:,i] -= box[i]*images[:,:,i] 
    print('pos',position.shape[0], np.amax(position[]), np.amin(position))
    
    
    num_times = position.shape[0]//every

    fluid_id = np.where((species[0,:] == 0))
    print('fluid id',fluid_id[0].shape)
    
    position = position[::every, fluid_id[0], :]
    print(position.shape)
    velocity = velocity[::every,fluid_id[0], :]

    step = 5
    dx = step*np.arange((slab)//step) + step + slab_start
    print(dx)
    
    vel_res = []
    dy_res = []
    viscosities = []
    for i in range(len(dx)):
        print(i)
        vel = []
        dy = []
        for t in range(num_times):
            #selecting fluid particles in the length of the pore (then select in the y direction)
            fluid_pos = position[t, np.logical_and( position[t,:, 0] > dx[i] - step, position[t,:, 0] < dx[i]), 1]
            #print( np.amax(fluid_pos), np.amin(fluid_pos))
            #speed in x direction from these particles
            speed = velocity[t, np.logical_and( position[t, :, 0] > dx[i] - step, position[t, :, 0] < dx[i]),0]
            #print(speed, np.amax(speed), np.amin(speed))
            vx, ygrid = yaxis(fluid_pos, speed)
            vel.append(vx)
            dy.append(ygrid)
        #print(np.array(vel).shape, np.array(dy).shape)
        vel = np.mean(np.array(vel), axis = 0)
        dy = np.mean(np.array(dy), axis = 0)
        #print(vel.shape, dy.shape)
        vel_res.append(vel)
        dy_res.append(dy)
        visc, cov = curve_fit(NS, dy[i], vel[i])
        viscosities.append(visc)
    print(viscosities)
    print(vel_res, dy_res)

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

    #ygrid = np.arange(-, , 0.5)
    
    #grid = np.linspace(- pore_yz/2 , pore_yz/2, num = 1000, endpoint = False)


    plt.plot(dy_res[0]-0.25, vel_res[0], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(dy_res[4]-.25, vel_res[4], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(dy_res[8]-0.25, vel_res[8], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(dy_res[12]-0.25, vel_res[12], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(dy_res[-1]-0.25, vel_res[-1], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
    #plt.plot(dy_res[-2], vel_res[-2], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')
   # plt.plot(dy_res[6], vel_res[6], '-',color='royalblue', linewidth=1.2,fillstyle='full', label = 'D25')

    plt.plot(dy_res[0]-0.25 , NS(dy_res[0], viscosities[0]) , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[0])
    #plt.plot(dy_res[4], NS(dy_res[4], viscosities[4])  , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[1])
    #plt.plot(dy_res[8], NS(dy_res[8], viscosities[8]) , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[2])
    #plt.plot(dy_res[12], NS(dy_res[12], viscosities[12]) , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[3])
    #plt.plot(dy_res[-1], NS(dy_res[-1], viscosities[-1]) , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[4])
    #plt.plot(dy_res[5], NS(dy_res[5], viscosities[5]) , '-', color='red',linewidth=1, linestyle='dashed', label = 'NS fit visc=%5.3f' % viscosities[5])


    plt.legend()

    plt.xlabel(r'$ y/ \sigma$')
    plt.ylabel(r'$ V_x/ \sqrt{\varepsilon / m} $') 
    #print(np.max(time0)-0.8, np.max(time1)-1,np.max(time2)-1.2)
    #source = 10
    plt.ylim([0,1])
    plt.xlim([0+sym[1],30+sym[1]])
    plt.legend(loc = 'upper left')
    #plt.axvline(x=source+sym, color='k', linestyle='--',linewidth=0.4)
 

    plt.axvspan(-pore_yz/2,pore_yz/2, alpha=0.5, color='grey')
   # plt.axvspan(0+sym, source+sym, alpha=0.5, color='gold')

    plt.savefig('vel.pdf')
   



if __name__ == '__main__':
    main()
