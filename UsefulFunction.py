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



"""useful functions"""


def reconstruct_signal(times, freqs, coeffs):
    """
    Reconstruct the signal in the time domain from the Fourier spectrum.
    
    Parameters:
    - times: array of time points where the signal is to be evaluated
    - freqs: array of frequencies (usually np.arange(0, N), where N is the number of frequencies)
    - coeffs: array of complex Fourier coefficients corresponding to the frequencies
    
    Returns:
    - signal: array of signal values at the specified time points
    """
    # Number of frequency bins
    N = len(freqs)
    
    # Initialize the signal array
    signal = np.zeros_like(times, dtype=complex)
    
    # Sum the contributions of each frequency component
    for i, f in enumerate(freqs):
        # Fourier components: exp(2 * pi * f * t)
        signal += coeffs[i] * np.exp(2j * np.pi * f * times)
    
    # The real part of the signal is the time-domain signal (since it's typically real-valued)
    return np.array(np.real(signal))



def generate_complex_array(r, length):
    """
    Generates an array of complex numbers with fixed magnitude r and random phases.
    
    Parameters:
    - r: The fixed radius (magnitude) of the complex numbers
    - length: The length of the output array
    
    Returns:
    - complex_array: An array of complex numbers with fixed magnitude r and random phases
    """
    # Generate random phases between 0 and 2*pi
    phases = np.random.uniform(0, 2 * np.pi, int(length))
    
    # Generate complex numbers in polar form (r, phase)
    complex_array = r * np.exp(1j * phases)
    
    return complex_array


def singal_uniform_in_k_space(max_freq, value, times): #random phases

    freqs = np.arange(0, max_freq)  # Frequencies from 0 to 99 Hz
    coeffs = generate_complex_array(value, (max_freq) )#with fixed radius
    magnitude = np.abs(coeffs)

    # Reconstruct the signal at the given time points
    signal = reconstruct_signal(times, freqs, coeffs)
   
    return signal

