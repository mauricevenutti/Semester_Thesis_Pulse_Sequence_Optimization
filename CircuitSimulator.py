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



"""Circuit Simulation"""

class Circuit_Simulator:
    
    
    # Zustände
    S1 = np.array([1, 0, 0, 1])
    S2 = np.array([0, 1, 1, 0])
    S3 = np.array([0, 0, 1, 0])
    S4 = np.array([0, 0, 0, 1])  # Ähnlich wie S3, aber andere Richtung
    S5 = np.array([0, 0, 0, 0])


    #s-space
    states = [S1,S2,S3,S4,S5]
    

    #Konstruktor
    def __init__(self,
                 Periodendauer,
                 N, #Anzahl Schaltungen in der Periode
                 data, #array [(zeitpunkt, Zustand), ()..., ()]
                 R, #Widerstand
                 L, #Induktivität
                ):
   

        self.data = data
        step_time=100@u_us
        self.T = Periodendauer
        self.N = N
        values = [[] for _ in range(4)]
        
        
        for i in range(4):
            for k in range(N):

                
                if(k>0):
                    values[i].append(((data[k][0]-1e-30)@u_ms, data[k-1][1][i]@u_V))
                    values[i].append((data[k][0]@u_ms, data[k][1][i]@u_V))
                else:
                    values[i].append((0.0@u_ms, 0@u_V))
                    values[i].append((1e-30@u_ms, data[k][1][i]@u_V))
            
                
                
        self.values = values
        self.cir = self.create_circuit(R, L)


        
        sim = self.cir.simulator(temperature = 25, nomial_temperature = 25)
        analysis = sim.transient(step_time=100@u_us, end_time= self.T@u_ms) #step time heisst nur wir wollen es mindestens so klein, die tatsächliche step time ist hier 1e-6 s
      

        self.time = np.array(analysis.time)
        self.current = np.array(analysis.L1)  
        
        
      
        new_time = np.arange(0, self.T + 1e-4, 1e-4)
        interpolation_func = interp1d(self.time, self.current, kind='linear', fill_value="extrapolate")
        new_current = interpolation_func(new_time)
        self.time = new_time
        self.current = new_current

        
        
        self.frequencies, self.fft_values, self.spectrum  = self.fourier_transform()

        

    
    '''create Circuit'''

    def create_circuit(self, Widerstand, Induktivität):
        cir = Circuit("Circuit")
        cir.V("input", "in", cir.gnd,12@u_V) 

        #add the switches
        cir.S(1, "in", "n1", "ctr1", cir.gnd, model = "SwitchModel1")
        cir.S(2, "in", "n2", "ctr2", cir.gnd, model = "SwitchModel2")
        cir.S(3, "n1", cir.gnd, "ctr3", cir.gnd, model = "SwitchModel3")
        cir.S(4, "n2", cir.gnd, "ctr4", cir.gnd, model = "SwitchModel4")

        #add diodes
        cir.model(
            'CustomDiode', 
            'D', 
            IS=1e-14,      # Saturation current
            N=1.0,         # Ideality factor
            RS=1e-2,       # Series resistance
            BV=50,         # Breakdown voltage
            IBV=1e-10,     # Reverse breakdown current
            CJO=1e-12,     # Junction capacitance
            VJ=0.7,        # Junction potential
            M=0.5          # Grading coefficient
        )
        cir.D(1, 'n1', 'in', model='CustomDiode')
        cir.D(2, 'n2', 'in', model='CustomDiode')
        cir.D(3, cir.gnd, 'n1', model='CustomDiode')
        cir.D(4, cir.gnd, 'n2', model='CustomDiode')
        
        for i in range(1,5):
            cir.model(f"SwitchModel{i}", "SW",Ron = 0.00001@u_mOhm, Roff =1e10@u_GOhm, Vt = 0.5 @u_V)
            cir.PieceWiseLinearVoltageSource(
            f"{i}__ctrl", f"ctr{i}", cir.gnd,
            values= self.values[i-1]
            )
        
        #add the coil
        cir.L(1, 'n1', 'n3', Induktivität@u_H)
        cir.R(1, 'n3', 'n2', Widerstand@u_Ohm)

        return cir
    
    

    def low_pass_filter(self, hf):
        
        dt = self.time[1] - self.time[0]
        sampling_rate =  1/dt
        nyquist = 0.5*sampling_rate

        normalized_cutoff = hf / nyquist
  
        b, a = sig.butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)

        # Apply the filter to the input signal
        filtered_signal = sig.lfilter(b, a, self.current)
        
        self.current = filtered_signal

        self.frequencies, self.fft_values, self.spectrum  = self.fourier_transform()
        
        

    def fourier_transform(self):
        # Calculate Fourier Transform
        dt = self.time[1] - self.time[0]  # Time step
        N = len(self.time)           # Number of points

        freq = np.fft.fftfreq(N, d=dt)  # Frequency axis

        fft_values = np.fft.fft(self.current)  # Fourier Transform
        
    
        # Only use the positive frequencies for plotting
        freq = freq[:N // 2]
        magnitude = np.abs(fft_values[:N // 2])  # Magnitude of FFT
    
        return freq, fft_values[:N//2], magnitude


    def make_plots(self, signal = None, save = False):
        
        threshold = 0.01 * np.max(self.spectrum)  

        valid_indices = np.where(self.spectrum > threshold)[0]  
        if len(valid_indices) > 0:
            max_freq = self.frequencies[valid_indices[-1]]  # Letzte gültige Frequenz
        else:
            max_freq = 0


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(self.time, self.current, label = "sim")
        if signal is not None:
            ax1.plot(self.time, signal, label = "desired signal")
            #ax1.plot(self.time, (self.current - signal)**2, label = "err^2")
        ax1.set_title("Current over Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Current through L1 (A)")
        ax1.legend()
        ax1.grid(True)
        
        
        ax2.plot(self.frequencies, self.spectrum, '.', label = "magnitudes simulation")
        if signal is not None:

            signal_fft = np.fft.fft(signal)
            sampling_rate = 1 / (self.time[1] - self.time[0])  # Sample rate (assuming uniform time steps)
            N = len(self.time)  # Number of points in the signal
            freqs = np.fft.fftfreq(N, d=self.time[1] - self.time[0])  # Frequency bins
            signal_fft_magnitude = np.abs(signal_fft)

            fft_mag = np.interp(self.frequencies, freqs, signal_fft_magnitude)
            ax2.plot(freqs,signal_fft_magnitude, '.', label = "magnitudes desired signal")
            
        ax2.set_title("Fourier Transform of Current Signal")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.set_xlim(0, max_freq)
        ax2.set_ylim(0, np.max(self.spectrum) * 1.5)
        ax2.legend()

        # Adjust layout to prevent overlap between subplots
        plt.tight_layout()
        
        if(save): plt.savefig("signal_and_spectrum.png")
        
        plt.show()
    
    
    def cost_function(self, signal):
  
        anz_switches = 0
        for i in range(1, len(self.data)):

            anz_switches += np.sum(self.data[i][1] != self.data[i-1][1])
        dt = self.time[1] - self.time[0]
        anz_switches = max(1, anz_switches)
        return np.sum(dt*(self.current - signal)**2), (self.current-signal)**2 #anz_switches

