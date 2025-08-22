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

from UsefulFunction import *

from CircuitSimulator import Circuit_Simulator

from data_generators_module import Data_Generators

from Plots_and_Cost import *
from pulse_pause_modulation import *

'''SIMMULATED ANNEALING'''
import math
def find_intervall(cf, data, global_params):
    '''cf ist die kostenfunktion'''

    R, L, U, period, highest_frequency = global_params
    time = np.arange(0, period/1000 + 1e-4, 1e-4)

    cf = np.array(cf)
    time = np.array(time)

    # Find the index of the maximum value in cf
    max_idx = np.argmax(cf)

    # Calculate total time span and 5% of the total time span
    total_time_span = time[-1] - time[0]
    delta = 0.05 * total_time_span

    # Determine the start and end times of the interval
    max_time = time[max_idx]
    start_time = 1000*max(max_time - delta, time[0])  # Ensure we don't go below the minimum time (in ms umrechnen)
    end_time = 1000*min(max_time + delta, time[-1])  # Ensure we don't exceed the maximum time (in mx umrechnen)

    #now find the indices for data
    a = 0
    b = len(data)-1

    
    for i in range(len(data)):
        if(data[i][0] > start_time):
            a = i
            break;
    for i in range(len(data)-1, -1, -1):

        if(data[i][0] < end_time):

            b = i
            break; 


            
    return a, b
    
def simmulated_annealing_data(signal, data_initial, max_iter, cooling_rate, global_parameters):
    """func: function to be approximated
        freq: freq of the function to be approximated"""
    
    current_state = np.copy(data_initial)
    current_cost, cf = cost_data(data_initial,signal, global_parameters)
    best_state = np.copy(current_state)
    best_cost = current_cost
    
    #finding the ideal T_init parameter, at T_init we want
    #arround 60-80% of the worse state to be accepted
    
    dc = 0
    for i in range(10):
        new_state = np.copy(current_state)
        index = random.randint(0,len(new_state) -1)
        new_state[index][1] = Circuit_Simulator.states[random.randint(0,4)]
        new_cost, dummy = cost_data(new_state, signal, global_parameters)
        delta_cost = abs(new_cost - current_cost)
        dc += delta_cost
    dc /= 10
    
    
    
    temperature = - dc/math.log(0.7)
    #print("T_init: ", temperature)

    costs = np.zeros((max_iter))
    for i in range(max_iter):
        
        new_state = np.copy(current_state)
        index = random.randint(0, len(data_initial)-1)
        if(i > 2*max_iter//3):
            a, b = find_intervall(cf, new_state, global_parameters) #suche das intervall, wo die approximation noch schlecht ist
            index = random.randint(a, b)

            if(i%100 == 0):
                print((a,b))
        
        new_state[index][1] = Circuit_Simulator.states[random.randint(0,4)]
        
        new_cost, cf = cost_data(new_state, signal, global_parameters)
        delta_cost = new_cost - current_cost
        #print(current_cost)
        if(delta_cost < 0 or random.random() < np.exp(-delta_cost / temperature)):
            current_state = new_state
            current_cost = new_cost
            
        
        if current_cost < best_cost:
            best_state = np.copy(current_state)
            best_cost = current_cost
        
        temperature *= cooling_rate
        costs[i] = best_cost
        if(i%100 == 0):
            print(best_cost)
            


       
 
        
    return best_state, costs