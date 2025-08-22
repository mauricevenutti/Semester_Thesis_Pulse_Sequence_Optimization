
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

from data_generators_module import Data_Generators

from Plots_and_Cost import *

# """ PULSE-PAUSE MODULATION """
#unterteile Zeit in N Abschnitte -> wir können N mal schalten

def data_pulse_pause(input, sign, global_params):
    
    '''
        period in ms,
        input: array of length N with number between 0 and 1  (sollte dem Absolutbetrag der gewünschten Steigung entsprechen)
        sign: +1 -> positive (S1,S4), -1 -> negative (S2, S3) (sollte Vorzeichem der Gewünschten Steigung entsprechen)
    '''    
    R, L, U, period, highest_frequency = global_params
    N = len(input)
    step = period/N
    data = []
    for i in range(0,N):

        if sign[i] > 0:
            on = 1
            off = 4
        else:
            on = 2
            off = 3
            
        data.append((i*step, Circuit_Simulator.states[on-1]))
        
        if(input[i] < 1.0) :
            data.append(((i*step + (input[i])*step), Circuit_Simulator.states[off-1]))

    return data


def ppm_joel(signal, time, N, global_params):
    """ func = desired signal
        N must be smaller than period/2 -> sonst verschiebt sich das signal nach rechts"""
    R, L, U, period, highest_frequency = global_params

    
    maximum = np.max(np.abs(signal))
    input = []
    sign = []
    n = len(time)
    step = max(1, n // (N)) # Calculate the step size, ensuring it's at least 1

    for i in range(N):
        anteil = signal[i]/maximum
        sign.append(np.sign(signal[i]))
        input.append(np.abs(anteil))

    return input, sign 

def ppm_maurice(signal, time, N, global_params):
    R, L, U, period, highest_frequency = global_params

    maximum = np.max(np.abs(signal))
    input = []
    sign = []

    for i in range(N-1):
        incline = (signal[i+1] - signal[i])/(time[i+1]-time[i])
        sign.append(np.sign(incline))
        anteil = np.min(np.array([(np.abs(incline) + R/L)/ (12 / L +  R/L), 1.0]))
        input.append(anteil)
    


    sign.append(np.sign(incline)) #bessere Lösung

    
    input.append(anteil) #bessere Lösung

    
    return input, sign
