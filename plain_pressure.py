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

def pressure(left_data,right_data,surface_area,box_length,sigmas):
    pressure_ = 0
    dimension = int((left_data.shape[1] - 2) / 2)
    v2_tot = np.zeros(dimension)
    p_par_par = np.zeros(dimension)
    p_obst_par = np.zeros(dimension)
    for i in range(dimension):
        v2_tot[i] += np.sum(np.power(left_data[:, dimension + i],2))
        v2_tot[i] += np.sum(np.power(right_data[:, dimension + i],2))

    v2_tot /= (5*surface_area)
    pressure_ += v2_tot

    for cur_right in right_data:
        for cur_left in left_data:
            r = np.zeros(dimension)
            for i in range(dimension):
                cur_dist = abs(cur_right[i] - cur_left[i])
                if (cur_dist > box_length[i] / 2):
                    cur_dist -= box_length[i]
                r[i] = cur_dist
            
            dist = np.linalg.norm(r)

            if (dist < 2.5):
                if (cur_left[-2] * cur_right[-2] < 0.5): # if one is 0 = particle
                    sigma = sigmas[0]
                    if (cur_left[-2] + cur_right[-2] > 0.5): # if one is 1 = obstacle
                        sigma = sigmas[1]
                        r_hat = r/dist
                        r_6 = (dist/sigma) ** (-6)
                        p_obst_par = (( (2.0 * r_6 * r_6) - r_6 ) * 24 / dist) * r_hat / surface_area
                    else:
                        r_hat = r/dist
                        r_6 = (dist/sigma) ** (-6)
                        p_par_par = (( (2.0 * r_6 * r_6) - r_6 ) * 24 / dist) * r_hat / surface_area
        pressure_ += p_obst_par + p_par_par

    return pressure_, v2_tot, p_obst_par, p_par_par, v2_tot+p_par_par


def pressure_par(data_tot):

    left_data = data_tot[0]
    right_data = data_tot[1]
    surface_area = data_tot[2]
    box_length = data_tot[3]
    sigmas = data_tot[4]

    pressure_ = 0
    dimension = int((left_data.shape[1] - 2) / 2)
    v2_tot = np.zeros(dimension)
    p_par_par = np.zeros(dimension)
    p_obst_par = np.zeros(dimension)
    for i in range(dimension):
        v2_tot[i] += np.sum(np.power(left_data[:, dimension + i],2))
        v2_tot[i] += np.sum(np.power(right_data[:, dimension + i],2))

    v2_tot /= (5*surface_area)
    pressure_ += v2_tot
   # print('p before',pressure_)

    for cur_right in right_data:
        for cur_left in left_data:
            r = np.zeros(dimension)
            for i in range(dimension):
                cur_dist = abs(cur_right[i] - cur_left[i])
                if (cur_dist > box_length[i] / 2):
                    cur_dist -= box_length[i]
                r[i] = cur_dist
            
            dist = np.linalg.norm(r)
            #print('dist',dist)

            if (dist < 2.5):
                if (cur_left[-2] * cur_right[-2] < 0.5): # if one is 0 = particle
                    sigma = sigmas[0]
                    if (cur_left[-2] + cur_right[-2] > 0.5): # if one is 1 = obstacle
                        sigma = sigmas[1]
                        r_hat = r/dist
                        r_6 = (dist/sigma) ** (-6)
                        p_obst_par += (( (2.0 * r_6 * r_6) - r_6 ) * 24 / dist) * r_hat / surface_area
                    else:
                        r_hat = r/dist
                        r_6 = (dist/sigma) ** (-6)
                        #print(r_6)
                        p_par_par = (( (2.0 * r_6 * r_6) - r_6 ) * 24 / dist) * r_hat / surface_area
                        #print(p_par_par)
                        p_par_par += p_par_par
    #print(  p_par_par)
    pressure_ += p_obst_par + p_par_par
    #print('p after', pressure_)

    return pressure_, v2_tot, p_obst_par, p_par_par, v2_tot+p_par_par

def main():
    box_length = [200,30,30]

    for m in range(1,2):
        H5 = h5py.File('result_25cst.h5')
        #pressure = H5['observables/pressure/value']
        #print("pressure=", np.mean(pressure[:]))
        Hpos = H5['particles/all/position']
        positions = np.array(Hpos['value'])
        Hvel = H5['particles/all/velocity']
        velocities = np.array(Hvel['value'])
        Hspecies = H5['particles/all/species']
        species = np.array(Hspecies['value'])
        Himages = H5['particles/all/image']
        images = np.array(Himages['value'])

        # bring particles back in box        
        for k in range(3):
            positions[:,:,k] -= box_length[k] * images[:,:,k]

        max_number_times = 3
        number_times = min(max_number_times,positions.shape[0])
        positions = positions[-number_times:,:,:]
        velocities = velocities[-number_times:,:,:]
        species = species[-number_times:,:]

        number_particles = positions.shape[1]
        dimensions = positions.shape[2]

        surface_area = 30 * 30

        # 0: particle_particle , 1: particle_obstacle
        sigmas = [1,0.95]

        complete_data = np.zeros([positions.shape[0],positions.shape[1],dimensions*2+2])
        complete_data[:,:,:dimensions] = positions
        complete_data[:,:,dimensions:dimensions * 2] = velocities
        complete_data[:,:,-2] = species
        complete_data[:,:,-1] = range(number_particles)
        

        #print(species.shape)
        time = []
        x_pos1 = np.arange(-97.5,-40,2.5)
        x_pos2 = np.arange(-40,40,0.5)
        x_pos3 = np.arange(40,97.5,2.5)
        x_pos = np.append(x_pos1 , x_pos2)
        
        x_pos = np.append(x_pos , x_pos3)
        #print(x_pos.shape, "*", x_pos)
        
        number_slabs = len(x_pos)
        print('%i number of slabs.\n%i number of timesteps.'%(number_slabs,number_times))
        
        final_results = np.zeros([number_slabs, 1 + 5 * dimensions])
        final_results[:,0] = x_pos

        for t in range(number_times):

            pressure_inputs = []

            for i in range(number_slabs):
                left_data = complete_data[t,np.logical_and(complete_data[t,:,0] > x_pos[i] - 2.5, complete_data[t,:,0] < x_pos[i]) ]
                right_data = complete_data[t,np.logical_and(complete_data[t,:,0] < x_pos[i] + 2.5, complete_data[t,:,0] > x_pos[i]) ]

                pressure_inputs.append([left_data, right_data, surface_area, box_length, sigmas])

            cur_results = 0

            with closing(Pool(processes=8)) as pool:
                cur_results = pool.map(pressure_par, pressure_inputs)
                pool.terminate()
            cur_results = np.array(cur_results)

            for i in range(5):
                final_results[:, 1 + i*dimensions : 1 + (i+1)*dimensions] += cur_results[:,i,:]

            print("%i percent passed."%((t+1)/number_times*100))

            print(len(left_data[t]),len(right_data[t]),number_slabs)
        final_results[:,1:] /= number_times
    banner_file = '#x_pos #P_tot_x #P_tot_y #P_tot_z #P_v2_x #P_v2_y #P_v2_z #P_obs_par_x #P_obs_par_y #P_obs_par_z #P_par_par_x #P_par_par_y #P_par_par_z #P_par_v2_x #P_par_v2_y #P_par_v2_z '
    np.savetxt("p_roya.dat", final_results, header = banner_file, comments = '')

if __name__ == '__main__':
    main()
