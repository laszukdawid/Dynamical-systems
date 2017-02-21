#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 21/02/2017
#
# Feel free to contact for any information.
#
# You can cite this code by referencing:
#   D. Laszuk, "Python implementation of Kuramoto systems," 2017-,
#   [Online] Available: http://www.laszukdawid.com/codes
#
# LICENCE:
# This program is free software on GNU General Public Licence version 3.
# For details of the copyright please see: http://www.gnu.org/licenses/.

from __future__ import print_function

import numpy as np
from scipy.integrate import odeint

__version__ = '0.1'
__author__ = 'Dawid Laszuk'

class Kuramoto(object):
    """
    Implementation of Kuramoto coupling model [1] with harmonic terms.
    It uses NumPy's arrays and Scipy `odeint` for numerical integration.
    
    Usage example:
    >>> kuramoto = Kuramoto(initial_values)
    >>> phase = kuramoto.solve(X)
    
    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
        (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """

    def __init__(self, init_values):
        """
        Passed arguments should be a dictionary with NumPy arrays
        for initial phase (Y0), intrisic frequencies (W) and coupling
        matrix (K). 
        """
        
        self.init_phase = np.array(init_values['Y0'])
        self.W = np.array(init_values['W'])
        self.K = np.array(init_values['K'])
        
    def kuramoto_ODE(self, y, t, w, k):
        """Function passed for ODE solver.
           In this case it is frequency Kuramoto model.
        """

        yt = y[:,None]
        dy = y-yt
        phase = w + np.sum(k*np.sin(dy),axis=1)
        
        return phase

    def solve(self, t):
        args = (self.W, self.K)
        phase = odeint(self.kuramoto_ODE, self.init_phase, t, args=args)
        return phase


######################################

if __name__ == "__main__":
    import pylab as plt
    
    ####################################################
    t0, t1, dt = 0, 40, 0.01
    T = np.arange(t0, t1, dt)

    # Y0, W, K are initial phase, intrisic freq and 
    # coupling K matrix respectively
    _Y0 = np.array([0, np.pi,0,1, 5, 2, 3])
    _W = np.array([28,19,11,9, 2, 4])
    _K = np.array([[ 2.3844,  1.2934,  0.6834,  2.0099,  1.9885],
                   [ 2.3854,  3.6510,  2.0467,  3.6252,  3.2463],
                   [ 1.1939,  4.4156,  1.1423,  0.2509,  4.1527],
                   [ 3.8386,  2.8487,  3.4895,  0.0683,  0.8246],
                   [ 3.9127,  1.2861,  2.9401,  0.1530,  0.6573]])

    # Preparing oscillators with Kuramoto model
    oscN = 5 # num of oscillators

    Y0 = _Y0[:oscN]
    W = _W[:oscN]
    K = _K[:oscN,:oscN]

    init_params = {'W':W, 'K':K, 'Y0':Y0}
    
    kuramoto = Kuramoto(init_params)
    odePhi = kuramoto.solve(T).T
    odeT = T[:-1]

    ##########################################
    # Plot the phases 
    plt.figure()

    for comp in range(len(W)):
        plt.subplot(len(W),1,comp+1)
        plt.plot(odeT, np.diff(odePhi[comp])/dt,'r')
        plt.ylabel('$\dot\phi_%i(t)$'%(comp+1))
    plt.savefig('phases')

    # Display plot
    plt.show()
