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



def simulation_plot_S(S, global_params):
    R, L, U, period, highest_frequency = global_params
    data = Data_Generators.data_equally_spaced(period, S)

    sim = Circuit_Simulator(period@u_ms, len(data) , data,R, L)
    
    if(not highest_frequency < 0):
        sim.low_pass_filter(highest_frequency)
    sim.make_plots()

def simmulation_plot_data(data, global_params):
    R, L, U, period, highest_frequency = global_params
    sim = Circuit_Simulator(period@u_ms, len(data) , data,R, L)

    if(not highest_frequency):
        sim.low_pass_filter(highest_frequency)
    sim.make_plots()


def cost(S, signal, global_params, make_plot = False):
    ''' S is the input array
        signal is a function - could also be an array -> update
    '''
    R, L, U, period, highest_frequency = global_params
    data = Data_Generators.data_equally_spaced(period, S)
    sim = Circuit_Simulator(period@u_ms, len(data) , data,R, L)
    
    if(not highest_frequency < 0):
        sim.low_pass_filter(highest_frequency)


    if make_plot:
        sim.make_plots(signal)
    return sim.cost_function(signal)


def cost_data(data, signal,global_params, make_plot = False):
    ''' S is the input array
        signal is a function - could also be an array -> update
    '''
    R, L, U, period, highest_frequency = global_params

    sim = Circuit_Simulator(period@u_ms,len(data),data,R, L)

    if(not highest_frequency < 0):
        sim.low_pass_filter(highest_frequency)

    if make_plot:
        sim.make_plots(signal)
    return sim.cost_function(signal)