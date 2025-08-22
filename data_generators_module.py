import numpy as np
import matplotlib.pyplot as plt

import PySpice
import logging

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

logger.setLevel(logging.CRITICAL)



from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from PySpice.Probe.Plot import plot
from scipy import signal as sig
import random

from UsefulFunction import reconstruct_signal, generate_complex_array, singal_uniform_in_k_space
from CircuitSimulator import Circuit_Simulator 

class Data_Generators:
    
    @staticmethod
    def data_equally_spaced(period, S):
        
        '''
            period in ms,
            S is an array length N with values from 1 to 5, should always start with 5 
            S[0] is ignored
        '''
        
        N = len(S)
        step = period/N
        data = []

        for i in range(0,N):
            data.append((i*step, Circuit_Simulator.states[S[i]-1]))
         
        
        return data
    
    @staticmethod
    def randomS(N):
        
        S = [5]
        for i in range(1,N):
            S.append(random.randint(1,5))

        return S
    
    @staticmethod
    def data_equally_spaced_random(period, N):
        
        '''
        period in ms
        N = number of Switches
        '''
        
        return Data_Generators.data_equally_spaced(period, Data_Generators.randomS(N))
    
    