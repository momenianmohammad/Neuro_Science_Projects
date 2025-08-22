# File: src/neural_simulator/__init__.py
"""
Biological Neural Network Simulator
===================================

A comprehensive toolkit for simulating biophysically realistic neural networks.

Main Components:
- Neuron models (Hodgkin-Huxley, LIF, AdEx)
- Synaptic models (AMPA, GABA, STDP)
- Network simulation and analysis
- Visualization tools
"""

from .models.hodgkin_huxley import HodgkinHuxleyNeuron
from .models.integrate_fire import LIFNeuron, AdExNeuron
from .networks.neural_network import NeuralNetwork
from .synapses.ampa_synapse import AMPASynapse
from .synapses.gaba_synapse import GABASynapse
from .synapses.stdp import STDPSynapse
from .analysis.spike_analysis import SpikeAnalysis
from .analysis.phase_plane import PhasePlaneAnalyzer
from .visualization.plotting import NetworkVisualizer

__version__ = "1.0.0"
__author__ = "Computational Neuroscience Lab"

# =============================================================================
# File: src/neural_simulator/models/hodgkin_huxley.py
"""
Hodgkin-Huxley Neuron Model
==========================

Implementation of the classic HH model with sodium, potassium, and leak currents.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class HodgkinHuxleyNeuron:
    """
    Hodgkin-Huxley neuron model with detailed ion channel dynamics.
    
    The model includes:
    - Voltage-gated sodium channels (transient)
    - Voltage-gated potassium channels (delayed rectifier)
    - Leak channels
    """
    
    def __init__(self, 
                 cm: float = 1.0,      # Membrane capacitance (μF/cm²)
                 gna: float = 120.0,   # Sodium conductance (mS/cm²)
                 gk: float = 36.0,     # Potassium conductance (mS/cm²)
                 gl: float = 0.3,      # Leak conductance (mS/cm²)
                 ena: float = 50.0,    # Sodium reversal potential (mV)
                 ek: float = -77.0,    # Potassium reversal potential (mV)
                 el: float = -54.387,  # Leak reversal potential (mV)
                 v_init: float = -65.0): # Initial membrane potential (mV)
        
        self.cm = cm
        self.gna = gna
        self.gk = gk
        self.gl = gl
        self.ena = ena
        self.ek = ek
        self.el = el
        self.v_init = v_init
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.v_init
        self.m = self.alpha_m(self.v_init) / (self.alpha_m(self.v_init) + self.beta_m(self.v_init))
        self.h = self.alpha_h(self.v_init) / (self.alpha_h(self.v_init) + self.beta_h(self.v_init))
        self.n = self.alpha_n(self.v_init) / (self.alpha_n(self.v_init) + self.beta_n(self.v_init))
    
    def alpha_m(self, v: float) -> float:
        """Sodium activation rate constant."""
        if abs(v + 40.0) < 1e-6:
            return 1.0
        return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
    
    def beta_m(self, v: float) -> float:
        """Sodium deactivation rate constant."""
        return 4.0 * np.exp(-(v + 65.0) / 18.0)
    
    def alpha_h(self, v: float) -> float:
        """Sodium inactivation rate constant."""
        return 0.07 * np.exp(-(v + 65.0) / 20.0)
    
    def beta_h(self, v: float) -> float:
        """Sodium de-inactivation rate constant."""
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    
    def alpha_n(self, v: float) -> float:
        """Potassium activation rate constant."""
        if abs(v + 55.0) < 1e-6:
            return 0.1
        return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
    
    def beta_n(self, v: float) -> float:
        """Potassium deactivation rate constant."""
        return 0.125 * np.exp(-(v + 65.0) / 80.0)
    
    def ionic_currents(self, v: float, m: float, h: float, n: float) -> Tuple[float, float, float]:
        """Calculate ionic currents."""
        i_na = self.gna * (m**3) * h * (v - self.ena)
        i_k = self.gk * (n**4) * (v - self.ek)
        i_l = self.gl * (v - self.el)
        return i_na, i_k, i_l
    
    def derivatives(self, v: float, m: float, h: float, n: float, i_ext: float) -> Tuple[float, float, float, float]:
        """Calculate derivatives for integration."""
        i_na, i_k, i_l = self.ionic_currents(v, m, h, n)
        
        dv_dt = (i_ext - i_na - i_k - i_l) / self.cm
        dm_dt = self.alpha_m(v) * (1 - m) - self.beta_m(v) * m
        dh_dt = self.alpha_h(v) * (1 - h) - self.beta_h(v) * h
        dn_dt = self.alpha_n(v) * (1 - n) - self.beta_n(v) * n
        
        return dv_dt, dm_dt, dh_dt, dn_dt
    
    def step(self, i_ext: float, dt: float):
        """Single integration step using RK4 method."""
        # RK4 integration
        k1_v, k1_m, k1_h, k1_n = self.derivatives(self.v, self.m, self.h, self.n, i_ext)
        
        k2_v, k2_m, k2_h, k2_n = self.derivatives(
            self.v + 0.5*dt*k1_v, self.m + 0.5*dt*k1_m, 
            self.h + 0.5*dt*k1_h, self.n + 0.5*dt*k1_n, i_ext)
        
        k3_v, k3_m, k3_h, k3_n = self.derivatives(
            self.v + 0.5*dt*k2_v, self.m + 0.5*dt*k2_m,
            self.h + 0.5*dt*k2_h, self.n + 0.5*dt*k2_n, i_ext)
        
        k4_v, k4_m, k4_h, k4_n = self.derivatives(
            self.v + dt*k3_v, self.m + dt*k3_m,
            self.h + dt*k3_h, self.n + dt*k3_n, i_ext)
        
        # Update state variables
        self.v += dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.m += dt/6 * (k1_m + 2*k2_m + 2*k3_m + k4_m)
        self.h += dt/6 * (k1_h + 2*k2_h + 2*k3_h + k4_h)
        self.n += dt/6 * (k1_n + 2*k2_n + 2*k3_n + k4_n)
    
    def simulate(self, current: List[float], dt: float = 0.01, duration: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Simulate neuron response to current injection.
        
        Parameters:
        -----------
        current : list - External current at each time step (μA/cm²)
        dt : float - Integration time step (ms)
        duration : float - Simulation duration (ms), if None uses len(current)*dt
        
        Returns:
        --------
        time : ndarray - Time vector (ms)
        voltage : ndarray - Membrane potential trace (mV)
        currents : dict - Ionic currents (μA/cm²)
        """
        if duration is None:
            duration = len(current) * dt
        
        n_steps = int(duration / dt)
        if len(current) < n_steps:
            current = current + [0.0] * (n_steps - len(current))
        
        # Initialize arrays
        time = np.linspace(0, duration, n_steps)
        voltage = np.zeros(n_steps)
        i_na_trace = np.zeros(n_steps)
        i_k_trace = np.zeros(n_steps)
        i_l_trace = np.zeros(n_steps)
        m_trace = np.zeros(n_steps)
        h_trace = np.zeros(n_steps)
        n_trace = np.zeros(n_steps)
        
        self.reset()
        
        # Simulate
        for i in range(n_steps):
            voltage[i] = self.v
            i_na, i_k, i_l = self.ionic_currents(self.v, self.m, self.h, self.n)
            i_na_trace[i] = i_na
            i_k_trace[i] = i_k
            i_l_trace[i] = i_l
            m_trace[i] = self.m
            h_trace[i] = self.h
            n_trace[i] = self.n
            
            self.step(current[i], dt)
        
        currents = {
            'i_na': i_na_trace,
            'i_k': i_k_trace,
            'i_l': i_l_trace,
            'i_ext': np.array(current[:n_steps]),
            'm': m_trace,
            'h': h_trace,
            'n': n_trace
        }
        
        return time, voltage, currents

# =============================================================================
# File: src/neural_simulator/models/integrate_fire.py
"""
Integrate-and-Fire Neuron Models
================================

Implementation of LIF and AdEx neuron models.
"""

import numpy as np
from typing import List, Tuple, Optional

class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron model.
    
    A simple yet effective model for large-scale network simulations.
    """
    
    def __init__(self,
                 tau_m: float = 20.0,    # Membrane time constant (ms)
                 v_rest: float = -65.0,  # Resting potential (mV)
                 v_thresh: float = -55.0, # Spike threshold (mV)
                 v_reset: float = -65.0,  # Reset potential (mV)
                 tau_ref: float = 2.0,    # Refractory period (ms)
                 r_m: float = 1.0,        # Membrane resistance (MΩ)
                 cm: float = 1.0):        # Membrane capacitance (nF)
        
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_ref = tau_ref
        self.r_m = r_m
        self.cm = cm
        
        self.reset()
    
    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.v_rest
        self.t_last_spike = -np.inf
        self.spike_times = []
    
    def step(self, i_ext: float, dt: float, t: float) -> bool:
        """
        Single integration step.
        
        Returns True if neuron spikes.
        """
        # Check refractory period
        if t - self.t_last_spike < self.tau_ref:
            return False
        
        # Update membrane potential
        dv_dt = (-(self.v - self.v_rest) + self.r_m * i_ext) / self.tau_m
        self.v += dv_dt * dt
        
        # Check for spike
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.t_last_spike = t
            self.spike_times.append(t)
            return True
        
        return False
    
    def simulate(self, current: List[float], dt: float = 0.1, duration: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Simulate LIF neuron response.
        
        Returns:
        --------
        time : ndarray - Time vector
        voltage : ndarray - Membrane potential trace
        spikes : list - Spike times
        """
        if duration is None:
            duration = len(current) * dt
        
        n_steps = int(duration / dt)
        if len(current) < n_steps:
            current = current + [0.0] * (n_steps - len(current))
        
        time = np.linspace(0, duration, n_steps)
        voltage = np.zeros(n_steps)
        
        self.reset()
        
        for i in range(n_steps):
            voltage[i] = self.v
            self.step(current[i], dt, time[i])
        
        return time, voltage, self.spike_times

class AdExNeuron:
    """
    Adaptive Exponential Integrate-and-Fire neuron model.
    
    Includes spike-rate adaptation and exponential approach to threshold.
    """
    
    def __init__(self,
                 cm: float = 1.0,         # Membrane capacitance (nF)
                 gl: float = 0.05,        # Leak conductance (μS)
                 el: float = -65.0,       # Leak reversal potential (mV)
                 vt: float = -55.0,       # Threshold potential (mV)
                 dt_spike: float = 2.0,   # Spike slope factor (mV)
                 v_reset: float = -65.0,  # Reset potential (mV)
                 tau_w: float = 100.0,    # Adaptation time constant (ms)
                 a: float = 0.0,          # Subthreshold adaptation (μS)
                 b: float = 0.1,          # Spike-triggered adaptation (nA)
                 v_spike: float = 20.0,   # Spike detection threshold (mV)
                 tau_ref: float = 2.0):   # Refractory period (ms)
        
        self.cm = cm
        self.gl = gl
        self.el = el
        self.vt = vt
        self.dt_spike = dt_spike
        self.v_reset = v_reset
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.v_spike = v_spike
        self.tau_ref = tau_ref
        
        self.reset()
    
    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.el
        self.w = 0.0
        self.t_last_spike = -np.inf
        self.spike_times = []
    
    def step(self, i_ext: float, dt: float, t: float) -> bool:
        """Single integration step."""
        # Check refractory period
        if t - self.t_last_spike < self.tau_ref:
            return False
        
        # Exponential term
        exp_term = self.dt_spike * np.exp((self.v - self.vt) / self.dt_spike)
        
        # Membrane equation
        dv_dt = (-self.gl * (self.v - self.el) + self.gl * exp_term - self.w + i_ext) / self.cm
        
        # Adaptation variable
        dw_dt = (self.a * (self.v - self.el) - self.w) / self.tau_w
        
        # Update variables
        self.v += dv_dt * dt
        self.w += dw_dt * dt
        
        # Check for spike
        if self.v >= self.v_spike:
            self.v = self.v_reset
            self.w += self.b
            self.t_last_spike = t
            self.spike_times.append(t)
            return True
        
        return False

# =============================================================================
# File: src/neural_simulator/synapses/ampa_synapse.py
"""
AMPA Synaptic Model
==================

Excitatory glutamatergic synapse model.
"""

import numpy as np
from typing import List

class AMPASynapse:
    """
    AMPA receptor-mediated excitatory synapse.
    
    Double exponential conductance model.
    """
    
    def __init__(self,
                 tau_rise: float = 0.2,   # Rise time constant (ms)
                 tau_decay: float = 2.0,  # Decay time constant (ms)
                 e_rev: float = 0.0,      # Reversal potential (mV)
                 weight: float = 1.0,     # Synaptic weight (nS)
                 delay: float = 1.0):     # Synaptic delay (ms)
        
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.e_rev = e_rev
        self.weight = weight
        self.delay = delay
        
        # Normalization factor for conductance
        tp = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        self.norm_factor = -1 / (np.exp(-tp/tau_decay) - np.exp(-tp/tau_rise))
        
        self.reset()
    
    def reset(self):
        """Reset synapse state."""
        self.g = 0.0  # Conductance
        self.r = 0.0  # Rise variable
        self.d = 0.0  # Decay variable
        self.spike_buffer = []  # Delayed spikes
    
    def add_spike(self, t: float):
        """Add presynaptic spike."""
        self.spike_buffer.append(t + self.delay)
    
    def step(self, dt: float, t: float, v_post: float) -> float:
        """
        Update synapse and return synaptic current.
        
        Parameters:
        -----------
        dt : float - Time step
        t : float - Current time
        v_post : float - Postsynaptic voltage
        
        Returns:
        --------
        i_syn : float - Synaptic current (pA)
        """
        # Process delayed spikes
        spikes_to_process = [spike_time for spike_time in self.spike_buffer if spike_time <= t]
        self.spike_buffer = [spike_time for spike_time in self.spike_buffer if spike_time > t]
        
        # Add spike contributions
        for _ in spikes_to_process:
            self.r += 1.0
            self.d += 1.0
        
        # Update kinetic variables
        self.r -= self.r * dt / self.tau_rise
        self.d -= self.d * dt / self.tau_decay
        
        # Calculate conductance
        self.g = self.weight * self.norm_factor * (self.d - self.r)
        
        # Calculate current
        i_syn = self.g * (v_post - self.e_rev)
        
        return i_syn

# =============================================================================
# File: src/neural_simulator/synapses/gaba_synapse.py
"""
GABA Synaptic Model
==================

Inhibitory GABAergic synapse model.
"""

import numpy as np

class GABASynapse:
    """
    GABA receptor-mediated inhibitory synapse.
    """
    
    def __init__(self,
                 tau_rise: float = 0.5,    # Rise time constant (ms)
                 tau_decay: float = 5.0,   # Decay time constant (ms)
                 e_rev: float = -70.0,     # Reversal potential (mV)
                 weight: float = 1.0,      # Synaptic weight (nS)
                 delay: float = 1.0):      # Synaptic delay (ms)
        
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.e_rev = e_rev
        self.weight = weight
        self.delay = delay
        
        # Normalization factor
        tp = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        self.norm_factor = -1 / (np.exp(-tp/tau_decay) - np.exp(-tp/tau_rise))
        
        self.reset()
    
    def reset(self):
        """Reset synapse state."""
        self.g = 0.0
        self.r = 0.0
        self.d = 0.0
        self.spike_buffer = []
    
    def add_spike(self, t: float):
        """Add presynaptic spike."""
        self.spike_buffer.append(t + self.delay)
    
    def step(self, dt: float, t: float, v_post: float) -> float:
        """Update synapse and return synaptic current."""
        # Process delayed spikes
        spikes_to_process = [spike_time for spike_time in self.spike_buffer if spike_time <= t]
        self.spike_buffer = [spike_time for spike_time in self.spike_buffer if spike_time > t]
        
        # Add spike contributions
        for _ in spikes_to_process:
            self.r += 1.0
            self.d += 1.0
        
        # Update kinetic variables
        self.r -= self.r * dt / self.tau_rise
        self.d -= self.d * dt / self.tau_decay
        
        # Calculate conductance and current
        self.g = self.weight * self.norm_factor * (self.d - self.r)
        i_syn = self.g * (v_post - self.e_rev)
        
        return i_syn

# =============================================================================
# File: src/neural_simulator/synapses/stdp.py
"""
Spike-Timing Dependent Plasticity
=================================

Implementation of STDP learning rule.
"""

import numpy as np
from typing import List

class STDPSynapse:
    """
    Spike-timing dependent plasticity synapse.
    
    Implements asymmetric Hebbian learning rule.
    """
    
    def __init__(self,
                 a_plus: float = 0.01,    # LTP amplitude
                 a_minus: float = 0.012,  # LTD amplitude
                 tau_plus: float = 20.0,  # LTP time constant (ms)
                 tau_minus: float = 20.0, # LTD time constant (ms)
                 w_min: float = 0.0,      # Minimum weight
                 w_max: float = 5.0):     # Maximum weight
        
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        
        self.reset()
    
    def reset(self):
        """Reset STDP variables."""
        self.x_pre = 0.0   # Presynaptic trace
        self.x_post = 0.0  # Postsynaptic trace
    
    def update_traces(self, dt: float, pre_spike: bool = False, post_spike: bool = False):
        """Update synaptic traces."""
        # Decay traces
        self.x_pre *= np.exp(-dt / self.tau_plus)
        self.x_post *= np.exp(-dt / self.tau_minus)
        
        # Add spikes
        if pre_spike:
            self.x_pre += 1.0
        if post_spike:
            self.x_post += 1.0
    
    def update_weight(self, weight: float, pre_spike: bool = False, post_spike: bool = False) -> float:
        """
        Update synaptic weight based on spike timing.
        
        Parameters:
        -----------
        weight : float - Current synaptic weight
        pre_spike : bool - Presynaptic spike occurred
        post_spike : bool - Postsynaptic spike occurred
        
        Returns:
        --------
        new_weight : float - Updated synaptic weight
        """
        dw = 0.0
        
        if pre_spike:
            # LTD: pre after post
            dw -= self.a_minus * self.x_post
        
        if post_spike:
            # LTP: post after pre
            dw += self.a_plus * self.x_pre
        
        # Update weight
        new_weight = np.clip(weight + dw, self.w_min, self.w_max)
        
        return new_weight

# =============================================================================
# File: src/neural_simulator/networks/neural_network.py
"""
Neural Network Simulation
=========================

Framework for simulating networks of interconnected neurons.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import networkx as nx

class Population:
    """Neural population container."""
    
    def __init__(self, name: str, neuron_class: Any, size: int, **neuron_params):
        self.name = name
        self.size = size
        self.neurons = [neuron_class(**neuron_params) for _ in range(size)]
        self.spike_times = [[] for _ in range(size)]
        self.voltages = []

class Connection:
    """Synaptic connection between populations."""
    
    def __init__(self, pre_pop: Population, post_pop: Population, 
                 synapse_class: Any, connection_matrix: np.ndarray, **synapse_params):
        self.pre_pop = pre_pop
        self.post_pop = post_pop
        self.connection_matrix = connection_matrix
        
        # Create synapses
        self.synapses = {}
        for i in range(pre_pop.size):
            for j in range(post_pop.size):
                if connection_matrix[i, j] > 0:
                    weight = synapse_params.get('weight', 1.0) * connection_matrix[i, j]
                    synapse_params['weight'] = weight
                    self.synapses[(i, j)] = synapse_class(**synapse_params)

class NeuralNetwork:
    """
    Neural network simulator.
    
    Supports multiple populations and connection types.
    """
    
    def __init__(self):
        self.populations = {}
        self.connections = []
        self.time = 0.0
        self.dt = 0.1
    
    def add_population(self, name: str, neuron_class: Any, size: int, **neuron_params) -> Population:
        """Add a population of neurons."""
        pop = Population(name, neuron_class, size, **neuron_params)
        self.populations[name] = pop
        return pop
    
    def connect_populations(self, pre_pop: Population, post_pop: Population,
                          synapse_class: Any, connection_prob: float = 0.1,
                          weight: float = 1.0, **synapse_params):
        """Connect two populations with given probability."""
        # Generate connection matrix
        connection_matrix = np.random.random((pre_pop.size, post_pop.size))
        connection_matrix = (connection_matrix < connection_prob).astype(float)
        
        # Create connection
        synapse_params['weight'] = weight
        conn = Connection(pre_pop, post_pop, synapse_class, connection_matrix, **synapse_params)
        self.connections.append(conn)
    
    def simulate(self, duration: float, dt: float = 0.1, 
                 external_current: float = 0.0, record_voltage: bool = True) -> Tuple[Dict, Dict]:
        """
        Simulate the network.
        
        Parameters:
        -----------
        duration : float - Simulation duration (ms)
        dt : float - Integration time step (ms)
        external_current : float - External current to all neurons (pA)
        record_voltage : bool - Whether to record voltage traces
        
        Returns:
        --------
        spikes : dict - Spike times for each population
        voltages : dict - Voltage traces (if recorded)
        """
        self.dt = dt
        n_steps = int(duration / dt)
        
        # Initialize recording
        spikes = {name: [[] for _ in range(pop.size)] for name, pop in self.populations.items()}
        voltages = {name: [] for name in self.populations.keys()} if record_voltage else {}
        
        # Reset all neurons and synapses
        for pop in self.populations.values():
            for neuron in pop.neurons:
                neuron.reset()
        
        for conn in self.connections:
            for synapse in conn.synapses.values():
                synapse.reset()
        
        # Simulation loop
        for step in range(n_steps):
            self.time = step * dt
            
            # Record voltages
            if record_voltage:
                for name, pop in self.populations.items():
                    voltages[name].append([neuron.v for neuron in pop.neurons])
            
            # Update neurons
            current_spikes = defaultdict(list)
            
            for name, pop in self.populations.items():
                for i, neuron in enumerate(pop.neurons):
                    # Calculate synaptic currents
                    i_syn = 0.0
                    for conn in self.connections:
                        if conn.post_pop == pop:
                            for j in range(conn.pre_pop.size):
                                if (j, i) in conn.synapses:
                                    synapse = conn.synapses[(j, i)]
                                    i_syn += synapse.step(dt, self.time, neuron.v)
                    
                    # Update neuron
                    total_current = external_current - i_syn
                    spike = neuron.step(total_current, dt, self.time)
                    
                    if spike:
                        current_spikes[name].append(i)
                        spikes[name][i].append(self.time)
            
            # Update synapses with spikes
            for conn in self.connections:
                pre_name = conn.pre_pop.name
                if pre_name in current_spikes:
                    for pre_idx in current_spikes[pre_name]:
                        for post_idx in range(conn.post_pop.size):
                            if (pre_idx, post_idx) in conn.synapses:
                                conn.synapses[(pre_idx, post_idx)].add_spike(self.time)
        
        # Convert voltage recordings to numpy arrays
        if record_voltage:
            for name in voltages:
                voltages[name] = np.array(voltages[name])
        
        return spikes, voltages
    
    def plot_raster(self, spikes: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Plot raster plot of network activity."""
        fig, ax = plt.subplots(figsize=figsize)
        
        y_offset = 0
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.populations)))
        
        for i, (name, pop_spikes) in enumerate(spikes.items()):
            pop_size = len(pop_spikes)
            
            for neuron_idx, spike_times in enumerate(pop_spikes):
                if spike_times:
                    ax.scatter(spike_times, [y_offset + neuron_idx] * len(spike_times),
                             s=1, c=[colors[i]], alpha=0.7)
            
            # Add population label
            ax.text(-0.02, y_offset + pop_size/2, name, 
                   transform=ax.get_yaxis_transform(), 
                   ha='right', va='center', fontweight='bold')
            
            y_offset += pop_size
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Network Raster Plot')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_population_rates(self, spikes: Dict, bin_size: float = 10.0, figsize: Tuple[int, int] = (12, 6)):
        """Plot population firing rates over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.populations)))
        
        for i, (name, pop_spikes) in enumerate(spikes.items()):
            # Collect all spike times
            all_spikes = []
            for spike_times in pop_spikes:
                all_spikes.extend(spike_times)
            
            if all_spikes:
                # Create histogram
                max_time = max(all_spikes)
                bins = np.arange(0, max_time + bin_size, bin_size)
                counts, _ = np.histogram(all_spikes, bins)
                
                # Convert to firing rate (Hz)
                rates = counts / (bin_size / 1000.0) / len(pop_spikes)
                time_centers = bins[:-1] + bin_size / 2
                
                ax.plot(time_centers, rates, label=name, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Population Firing Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# File: src/neural_simulator/analysis/spike_analysis.py
"""
Spike Train Analysis Tools
==========================

Tools for analyzing spike trains and neural activity patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import List, Dict, Tuple, Optional

class SpikeAnalysis:
    """Comprehensive spike train analysis toolkit."""
    
    @staticmethod
    def firing_rate(spike_times: List[float], duration: float, bin_size: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate instantaneous firing rate.
        
        Parameters:
        -----------
        spike_times : list - Spike times (ms)
        duration : float - Total duration (ms)
        bin_size : float - Bin size for rate calculation (ms)
        
        Returns:
        --------
        time_bins : ndarray - Time bin centers (ms)
        rates : ndarray - Firing rates (Hz)
        """
        if not spike_times:
            bins = np.arange(0, duration, bin_size)
            return bins + bin_size/2, np.zeros(len(bins))
        
        bins = np.arange(0, duration + bin_size, bin_size)
        counts, _ = np.histogram(spike_times, bins)
        rates = counts / (bin_size / 1000.0)  # Convert to Hz
        time_bins = bins[:-1] + bin_size / 2
        
        return time_bins, rates
    
    @staticmethod
    def isi_analysis(spike_times: List[float]) -> Dict[str, float]:
        """
        Analyze interspike intervals.
        
        Returns:
        --------
        stats : dict - ISI statistics
        """
        if len(spike_times) < 2:
            return {'mean_isi': np.nan, 'cv_isi': np.nan, 'median_isi': np.nan}
        
        isis = np.diff(spike_times)
        
        stats = {
            'mean_isi': np.mean(isis),
            'std_isi': np.std(isis),
            'cv_isi': np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else np.nan,
            'median_isi': np.median(isis),
            'min_isi': np.min(isis),
            'max_isi': np.max(isis)
        }
        
        return stats
    
    @staticmethod
    def cross_correlation(spikes1: List[float], spikes2: List[float], 
                         max_lag: float = 100.0, bin_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cross-correlation between two spike trains.
        
        Parameters:
        -----------
        spikes1, spikes2 : list - Spike times (ms)
        max_lag : float - Maximum lag for correlation (ms)
        bin_size : float - Bin size (ms)
        
        Returns:
        --------
        lags : ndarray - Lag times (ms)
        correlation : ndarray - Cross-correlation values
        """
        # Create binary spike trains
        max_time = max(max(spikes1) if spikes1 else [0], max(spikes2) if spikes2 else [0])
        n_bins = int(max_time / bin_size) + 1
        
        train1 = np.zeros(n_bins)
        train2 = np.zeros(n_bins)
        
        for spike in spikes1:
            bin_idx = int(spike / bin_size)
            if bin_idx < n_bins:
                train1[bin_idx] = 1
        
        for spike in spikes2:
            bin_idx = int(spike / bin_size)
            if bin_idx < n_bins:
                train2[bin_idx] = 1
        
        # Calculate cross-correlation
        correlation = signal.correlate(train1, train2, mode='full')
        lags = signal.correlation_lags(len(train1), len(train2)) * bin_size
        
        # Restrict to desired lag range
        lag_mask = np.abs(lags) <= max_lag
        lags = lags[lag_mask]
        correlation = correlation[lag_mask]
        
        return lags, correlation
    
    @staticmethod
    def burst_detection(spike_times: List[float], max_isi: float = 10.0, 
                       min_spikes: int = 3) -> List[Tuple[float, float, int]]:
        """
        Detect bursts in spike train.
        
        Parameters:
        -----------
        spike_times : list - Spike times (ms)
        max_isi : float - Maximum ISI within burst (ms)
        min_spikes : int - Minimum spikes per burst
        
        Returns:
        --------
        bursts : list - List of (start_time, end_time, n_spikes) tuples
        """
        if len(spike_times) < min_spikes:
            return []
        
        spikes = np.array(sorted(spike_times))
        isis = np.diff(spikes)
        
        # Find burst boundaries
        burst_starts = []
        burst_ends = []
        in_burst = False
        burst_start_idx = 0
        
        for i, isi in enumerate(isis):
            if isi <= max_isi:
                if not in_burst:
                    in_burst = True
                    burst_start_idx = i
            else:
                if in_burst:
                    burst_length = i - burst_start_idx + 1
                    if burst_length >= min_spikes:
                        burst_starts.append(spikes[burst_start_idx])
                        burst_ends.append(spikes[i])
                    in_burst = False
        
        # Check final burst
        if in_burst:
            burst_length = len(spikes) - burst_start_idx
            if burst_length >= min_spikes:
                burst_starts.append(spikes[burst_start_idx])
                burst_ends.append(spikes[-1])
        
        bursts = [(start, end, int((end - start) / np.mean(isis)) + 1) 
                 for start, end in zip(burst_starts, burst_ends)]
        
        return bursts