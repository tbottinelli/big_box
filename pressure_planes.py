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
from email import header
from xml.etree.ElementTree import Comment
import h5py
import os
from numpy import *
#from pylab import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Pool
from contextlib import closing
from scipy.spatial.distance import cdist
import math

def pressure_par(data_tot):

    left_data = data_tot[0]
    right_data = data_tot[1]
    areas = data_tot[2]
    box_length = data_tot[3]
    sigmas = data_tot[4]
    epsilon = data_tot[5]
    cutoff = data_tot[6]
    
    slab_x = 2.5

    pressure_ = 0
    dimension = int((left_data.shape[1] - 2) / 2)
    v2_tot = np.zeros(dimension)
    for i in range(dimension):
        v2_tot[i] += np.sum(np.power(left_data[:, dimension + i],2))
        v2_tot[i] += np.sum(np.power(right_data[:, dimension + i],2))
    v2_tot /= (2*slab_x*areas)
    pressure_ += v2_tot
    #print('pressure before loop', pressure_)

    #create species array that accounts for f-f/f-p interaction
    species_sum = np.array( left_data[None,:,-2] + right_data[:,None, -2] )
    #print('species matrix',species_sum.shape) #rightxleft
    ff_id = np.array( np.where((species_sum == 0)) )
    pf_id = np.array( np.where((species_sum == 1)) )
    #print('id fluid',ff_id)
    ff_max_r = np.amax(ff_id[0])+1
    ff_max_l = np.amax(ff_id[1])+1

    #print('id_max min', ff_max_r, ff_max_l)

    parameters = [sigmas, epsilon, cutoff]
    #create sigmas,epsilon,cutoff arrays
    ff = np.array( [np.full((ff_max_r,ff_max_l), parameters[i][0]) for i in range(len(parameters)) ] )
    fp = np.array( [np.full((ff_max_r, species_sum.shape[1] - ff_max_l), parameters[i][1]) for i in range(len(parameters)) ] )
    pf = np.array( [np.full((species_sum.shape[0] - ff_max_r, ff_max_l), parameters[i][1]) for i in range(len(parameters)) ])
    pp = np.array( [np.full((species_sum.shape[0] - ff_max_r, species_sum.shape[1] - ff_max_l), parameters[i][2]) for i in range(len(parameters)) ] )
    
    block = [np.block([ [ff[i] ,fp[i]], [pf[i], pp[i]] ])for i in range(len(parameters)) ]
    sigma = block[0]
    epsilon = block[1]
    cutoff = block[2]
    #print('block', np.array_equal(species_sum.shape, sigma.shape)) #true!
    #print(sigma, epsilon, cutoff)
   
    #create distance array
    cur_dist_x = abs(np.subtract.outer(right_data[:,0], left_data[:,0]))
    cur_dist_y = abs( np.subtract.outer(right_data[:,1], left_data[:,1]))
    cur_dist_z = abs( np.subtract.outer(right_data[:,2], left_data[:,2]))
    #print('dist x',np.array_equal(species_sum.shape, cur_dist_x.shape)) #true
    
    cur_dist = np.stack((cur_dist_x, cur_dist_y, cur_dist_z), axis = 0)
    #print('cur dist tot',cur_dist.shape)
    cur_dist1 = np.array( [  np.where( cur_dist[i,:,:] > box_length[i]/2, cur_dist[i,:,:] - box_length[i], cur_dist[i,:,:]  ) for i in range(dimension) ]) 
    #print(np.array_equal(cur_dist.shape, cur_dist1.shape)) #false but shape true, cur_dist1 is needed
    
    #norm of distances between left_right pair
    dist_norm = np.linalg.norm(cur_dist1, axis = 0)
    #print('dist norm shape', dist_norm) #mxn
 
    #keep smaller distances
    dist = np.where((dist_norm < cutoff), dist_norm, np.NaN)
    #print('distmin', dist) #dist values look the same

    r_hat = np.array(cur_dist1/dist) 
    #print('r_hat shape',r_hat[~np.isnan(r_hat)],r_hat.shape)
    r6 = np.array(( dist/sigma )**(-6))
    #print('dist sigma', r6[~np.isnan(r6)])
    pressure = (( (2.0 * r6**2) - r6 ) * 24 * epsilon / dist) * r_hat / areas
    #print('p shape',pressure[~np.isnan(pressure)])
    p = np.array([ np.nansum(pressure[i, :, :]) for i in range(3) ])
    #print('p', p, p.shape)

    p_ff = np.array([ np.nansum(pressure[i, :ff.shape[0], :ff.shape[1]]) for i in range(3) ])
    p_pf = np.array([ np.nansum(pressure[i, :fp.shape[0], fp.shape[1]:]) for i in range(3)]) + np.array([ np.nansum(pressure[i, pf.shape[0]:, :pf.shape[1]]) for i in range(3) ])
    #print('p_ff, p_pf', p_ff, p_pf)

    pressure_tot = pressure_ + p
    #print('pressure after for loops', pressure_tot)

    return pressure_tot, v2_tot, p_pf, p_ff, v2_tot + p_ff

def main():

    parser = argparse.ArgumentParser(prog = "pressure_planes.py")
    parser.add_argument('input', metavar = 'INPUT')
    args = parser.parse_args()

    H5 = h5py.File(args.input)

    box_length = np.diagonal(H5['particles/all/box/edges'])
    surface_area = box_length[1] * box_length[2]
    sigmas = [1,0.95,0.9] #ff/fp/pp
    epsilon = [1,1,0]
    cutoff = [2.5, 2.5,2.5]
    source_len = 10
    slab_len = 0.4*box_length[0]
    sym = -box_length[0]/2
    dx = 2.5
    pore_area = (15-sigmas[1])**2

    Hpos = H5['particles/all/position']
    positions = np.array(Hpos['value'])

    every = 2
    number_times = positions.shape[0] // every

    positions = np.array(Hpos['value'])[::every,:,:]
    Hvel = H5['particles/all/velocity']
    velocities = np.array(Hvel['value'])[::every,:,:]
    Hspecies = H5['particles/all/species']
    species = np.array(Hspecies['value'])[::every,:]
    Himages = H5['particles/all/image']
    images = np.array(Himages['value'])[::every,:,:]

    # bring particles back in box        
    for k in range(3):
        positions[:,:,k] -= box_length[k] * images[:,:,k]
    print(np.amin(positions[0,:,:]))
    number_particles = positions.shape[1]
    dimensions = positions.shape[2]
    print('dim', dimensions)

    complete_data = np.zeros([positions.shape[0],positions.shape[1],dimensions*2+2])
    complete_data[:,:,:dimensions] = positions
    complete_data[:,:,dimensions:dimensions * 2] = velocities
    complete_data[:,:,-2] = species
    complete_data[:,:,-1] = range(number_particles)
    
    print('comp data shape', complete_data.shape)

    #print(species.shape)
    time = []
    x_pos = np.arange(sym + dx, box_length[0] + sym , dx)
    #x_pos = np.concatenate( (x_pos1 , x_pos2, x_pos3), axis = None )
    print(x_pos.shape, "*", x_pos)
    number_slabs = len(x_pos)
    print('%i number of slabs.\n%i number of timesteps.'%(number_slabs,number_times))
    
    areas = int((box_length[0]-slab_len)/(2*dx) )*[surface_area]
    areas += int(slab_len/dx)*[pore_area]
    areas +=  int((box_length[0]-slab_len)/(2*dx) )*[surface_area]
    print(areas, len(areas))

    final_results = np.zeros([number_slabs, 1 + 5 * dimensions])
    print('final_result shape', final_results.shape)
    final_results[:,0] = x_pos

    for t in range(number_times):

        p_inputs = []
        for i in range(number_slabs):
            left_data = np.array( complete_data[t,np.logical_and(complete_data[t,:,0] > x_pos[i] - dx, complete_data[t,:,0] < x_pos[i]) ] )
            right_data = np.array( complete_data[t,np.logical_and(complete_data[t,:,0] < x_pos[i] + dx, complete_data[t,:,0] > x_pos[i]) ] )
           # print('left data', left_data[:,-2])
           # print('len right data', right_data[:,-2])

            p_inputs.append([left_data, right_data, areas[i], box_length, sigmas, epsilon, cutoff])
       # print('press_input', len(p_inputs))print('pot pf',potential_pf)
        cur_results = 0
        with closing(Pool(processes=8) ) as pool:
            cur_results = pool.map(pressure_par, p_inputs)
            pool.terminate()
        cur_results = np.array(cur_results)

        for i in range(5):
            final_results[:, 1 + i*dimensions : 1 + (i+1)*dimensions] += cur_results[:,i,:]

        print("%i percent passed."%((t+1)/number_times*100))

    final_results[:,1:] /= number_times
    banner_file = '#x_pos #P_tot_x #P_tot_y #P_tot_z #P_v2_x #P_v2_y #P_v2_z #P_obs_par_x #P_obs_par_y #P_obs_par_z #P_par_par_x #P_par_par_y #P_par_par_z #P_par_v2_x #P_par_v2_y #P_par_v2_z '
    #np.savetxt("tst15cst.dat", final_results, header = banner_file, comments = '')

if __name__ == '__main__':
    main()
