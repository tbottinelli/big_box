#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def NS_cyl(dy, visc):
    deltax = 80
    dp_25 = 2.440 - 4.097
    dp_15 = 11.0995 - 15.927
    dpdx = [dp_15/deltax, dp_25/deltax]
    R = [15/2, 25/2]
    
    return (-(dpdx[0]*R[0]**2)/(4*visc))*(1-(dy/R[0])**2)

#there is no difference between the two equations for cylinder and slit, also for other equations in other articles it's the same
def NS_slit(dy, visc):
    deltax = 80
    dp_25 = 2.440 - 4.097
    dp_15 = 11.0995 - 15.927
    dpdx = [dp_15/deltax, dp_25/deltax]
    h = [15, 25]
    
    return (dpdx[1]/(2*visc))*((dy**2)-(h[1]/2)**2)

    
def main():

    plt.rc('font', **{ 'family':'serif', 'serif': ['ptm'], 'size': 12 })
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=(r'\usepackage{textcomp}', r'\usepackage{amsmath}', r'\usepackage[T1]{fontenc}', r'\usepackage{times}', r'\usepackage{txfonts}'))
    plt.rc('legend', frameon=False, numpoints=1, fontsize=8, labelspacing=0.2, handlelength=2, handletextpad=0.5, borderaxespad=0.5)
    plt.rc('figure', figsize=(4.7,2))
    plt.rc('xtick', direction = 'in', top=True)
    plt.rc('ytick', direction = 'in', right=True)
    plt.rc('xtick.minor', visible = True, top=True)
    plt.rc('ytick.minor', visible = True, right=True)
    plt.rc('axes', linewidth = 0.7)
    plt.rc('lines', linewidth = 1, markersize = 2, markeredgewidth = 0)
    plt.rc('savefig', bbox = 'tight', pad_inches = 0.05, dpi = 600, transparent = False)
    plt.rc('ps', usedistiller = 'xpdf')

    dat25 = np.loadtxt('vel_25cst.dat', delimiter = ',')
    dat15 = np.loadtxt('vel_15cst.dat', delimiter = ',')
    dy15 = dat15[0]+0.25
    dy25 = dat25[0]+0.25

    #visc, cov = curve_fit(NS_cyl, dat15[0], dat15[1])
    visc_slit, cov = curve_fit(NS_slit, dat25[0], dat25[1])


    plt.plot(dy25, dat25[1],'-', color = 'deepskyblue', linewidth = 1.2, fillstyle = 'full', label = 'D25')
    plt.plot(dy25, NS_slit(dy25, visc_slit) ,'-', color = 'red', linewidth = 0.8, linestyle = 'dashed', label = 'NS fit viscosity = %5.3f' % visc_slit)

    plt.legend()
    plt.xlabel(r'$ y/ \sigma$')
    plt.ylabel(r'$ V_x/ \sqrt{\epsilon / m} $')
    plt.ylim([0,1.5])
    plt.xlim([0+15,30 + 15])
    plt.xlim([0-15,30 - 15])
    plt.legend(loc = 'upper left')
    plt.axvspan(-15/2, 15/2, alpha = 0.5, color = 'grey')
    plt.savefig('vel25slit.pdf')
    

if __name__== '__main__':
    main()


