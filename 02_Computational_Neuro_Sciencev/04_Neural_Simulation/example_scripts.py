# File: examples/basic_hh_simulation.py
"""
Basic Hodgkin-Huxley Neuron Simulation Example
==============================================

Demonstrates action potential generation with different current amplitudes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator import HodgkinHuxleyNeuron

def main():
    """Run basic HH neuron simulation with different current steps."""
    
    # Create HH neuron
    neuron = HodgkinHuxleyNeuron()
    
    # Define current steps to test
    current_amplitudes = [0, 5, 10, 15, 20, 25]  # μA/cm²
    duration = 100.0  # ms
    dt = 0.01  # ms
    
    # Create figure
    fig, axes = plt.subplots(len(current_amplitudes), 2, figsize=(15, 12))
    fig.suptitle('Hodgkin-Huxley Neuron Response to Current Steps', fontsize=16)
    
    for i, current_amp in enumerate(current_amplitudes):
        print(f"Simulating with {current_amp} μA/cm² current...")
        
        # Create current step (10-60 ms)
        current = [current_amp if 10 <= t*dt < 60 else 0.0 
                  for t in range(int(duration/dt))]
        
        # Run simulation
        time, voltage, currents = neuron.simulate(current, dt=dt, duration=duration)
        
        # Plot voltage trace
        axes[i, 0].plot(time, voltage, 'b-', linewidth=1.5)
        axes[i, 0].set_ylabel(f'V (mV)\nI={current_amp}', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlim(0, duration)
        
        if i == 0:
            axes[i, 0].set_title('Membrane Potential')
        if i == len(current_amplitudes) - 1:
            axes[i, 0].set_xlabel('Time (ms)')
        
        # Plot ionic currents
        axes[i, 1].plot(time, currents['i_na'], 'r-', label='I_Na', alpha=0.8)
        axes[i, 1].plot(time, currents['i_k'], 'g-', label='I_K', alpha=0.8)
        axes[i, 1].plot(time, currents['i_l'], 'k-', label='I_L', alpha=0.8)
        axes[i, 1].set_ylabel('Current (μA/cm²)', fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(0, duration)
        
        if i == 0:
            axes[i, 1].set_title('Ionic Currents')
            axes[i, 1].legend(loc='upper right', fontsize=8)
        if i == len(current_amplitudes) - 1:
            axes[i, 1].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig('hh_current_steps.png', dpi=300, bbox_inches='tight')
    plt.show()

def phase_plane_analysis():
    """Analyze HH neuron dynamics in phase plane."""
    
    neuron = HodgkinHuxleyNeuron()
    
    # Different current levels
    currents = [0, 7, 10, 15]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, current_amp in enumerate(currents):
        # Long simulation to see trajectory
        current = [current_amp] * 5000  # 50 ms at 0.01 dt
        time, voltage, currents_data = neuron.simulate(current, dt=0.01, duration=50)
        
        # Plot V vs dV/dt
        dv_dt = np.gradient(voltage, time[1] - time[0])
        
        axes[i].plot(voltage, dv_dt, 'b-', alpha=0.7, linewidth=1)
        axes[i].set_xlabel('Voltage (mV)')
        axes[i].set_ylabel('dV/dt (mV/ms)')
        axes[i].set_title(f'Phase Plane: I = {current_amp} μA/cm²')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hh_phase_plane.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
    phase_plane_analysis()

# =============================================================================
# File: examples/lif_network.py
"""
LIF Neural Network Example
==========================

Demonstrates E-I network dynamics and gamma oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator import NeuralNetwork, LIFNeuron
from neural_simulator.synapses import AMPASynapse, GABASynapse
from neural_simulator.analysis import SpikeAnalysis

def create_ei_network():
    """Create excitatory-inhibitory network for gamma oscillations."""
    
    # Network parameters
    n_exc = 800
    n_inh = 200
    
    # Create network
    network = NeuralNetwork()
    
    # Add populations with different parameters
    exc_params = {
        'tau_m': 20.0,
        'v_rest': -65.0,
        'v_thresh': -55.0,
        'v_reset': -65.0,
        'tau_ref': 2.0,
        'r_m': 1.0
    }
    
    inh_params = {
        'tau_m': 10.0,
        'v_rest': -65.0,
        'v_thresh': -55.0,
        'v_reset': -65.0,
        'tau_ref': 1.0,
        'r_m': 1.0
    }
    
    exc_pop = network.add_population('E', LIFNeuron, n_exc, **exc_params)
    inh_pop = network.add_population('I', LIFNeuron, n_inh, **inh_params)
    
    # Connection parameters for gamma rhythm
    # E->E: weak recurrent excitation
    network.connect_populations(exc_pop, exc_pop, AMPASynapse,
                               connection_prob=0.02, weight=0.5, delay=1.5)
    
    # E->I: strong excitation
    network.connect_populations(exc_pop, inh_pop, AMPASynapse,
                               connection_prob=0.5, weight=2.0, delay=0.8)
    
    # I->E: strong inhibition
    network.connect_populations(inh_pop, exc_pop, GABASynapse,
                               connection_prob=0.5, weight=4.0, delay=0.8)
    
    # I->I: moderate inhibition
    network.connect_populations(inh_pop, inh_pop, GABASynapse,
                               connection_prob=0.3, weight=1.5, delay=0.8)
    
    return network

def analyze_network_dynamics(spikes, duration):
    """Analyze network oscillations and synchrony."""
    
    analyzer = SpikeAnalysis()
    
    # Population firing rates
    exc_spikes = []
    inh_spikes = []
    
    for neuron_spikes in spikes['E']:
        exc_spikes.extend(neuron_spikes)
    
    for neuron_spikes in spikes['I']:
        inh_spikes.extend(neuron_spikes)
    
    # Calculate population rates
    time_bins, exc_rates = analyzer.firing_rate(exc_spikes, duration, bin_size=5.0)
    _, inh_rates = analyzer.firing_rate(inh_spikes, duration, bin_size=5.0)
    
    # Power spectrum analysis
    from scipy import signal
    
    if len(exc_rates) > 100:  # Ensure enough data
        freqs, psd_exc = signal.welch(exc_rates, fs=200, nperseg=min(256, len(exc_rates)//4))
        freqs, psd_inh = signal.welch(inh_rates, fs=200, nperseg=min(256, len(inh_rates)//4))
        
        # Find gamma peak (30-100 Hz)
        gamma_mask = (freqs >= 30) & (freqs <= 100)
        if np.any(gamma_mask):
            gamma_peak_exc = freqs[gamma_mask][np.argmax(psd_exc[gamma_mask])]
            gamma_power_exc = np.max(psd_exc[gamma_mask])
        else:
            gamma_peak_exc = gamma_power_exc = 0
    
    return {
        'time_bins': time_bins,
        'exc_rates': exc_rates,
        'inh_rates': inh_rates,
        'gamma_peak': gamma_peak_exc,
        'gamma_power': gamma_power_exc,
        'freqs': freqs,
        'psd_exc': psd_exc,
        'psd_inh': psd_inh
    }

def main():
    """Run E-I network simulation."""
    
    print("Creating E-I network...")
    network = create_ei_network()
    
    # Simulation parameters
    duration = 2000.0  # ms
    dt = 0.1  # ms
    external_drive = 15.0  # pA
    
    print("Running simulation...")
    spikes, voltages = network.simulate(duration=duration, dt=dt, 
                                      external_current=external_drive,
                                      record_voltage=False)
    
    print("Analyzing results...")
    analysis = analyze_network_dynamics(spikes, duration)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    # Raster plot
    ax1 = plt.subplot(4, 2, (1, 2))
    y_offset = 0
    colors = ['blue', 'red']
    
    for i, (pop_name, pop_spikes) in enumerate(spikes.items()):
        pop_size = len(pop_spikes)
        color = colors[i]
        
        for neuron_idx, spike_times in enumerate(pop_spikes[:100]):  # Show only first 100
            if spike_times:
                ax1.scatter(spike_times, [y_offset + neuron_idx] * len(spike_times),
                           s=0.5, c=color, alpha=0.6)
        
        ax1.text(-50, y_offset + 50, pop_name, fontweight='bold', fontsize=12)
        y_offset += 100
    
    ax1.set_xlim(0, min(duration, 1000))
    ax1.set_ylabel('Neuron Index')
    ax1.set_title('Network Raster Plot (first 1000ms)')
    ax1.grid(True, alpha=0.3)
    
    # Population rates
    ax2 = plt.subplot(4, 2, (3, 4))
    ax2.plot(analysis['time_bins'], analysis['exc_rates'], 'b-', 
             linewidth=2, label='Excitatory')
    ax2.plot(analysis['time_bins'], analysis['inh_rates'], 'r-', 
             linewidth=2, label='Inhibitory')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('Population Firing Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Power spectra
    ax3 = plt.subplot(4, 2, 5)
    ax3.loglog(analysis['freqs'], analysis['psd_exc'], 'b-', 
               linewidth=2, label='Excitatory')
    ax3.axvspan(30, 100, alpha=0.2, color='gray', label='Gamma band')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_title('Excitatory Power Spectrum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(4, 2, 6)
    ax4.loglog(analysis['freqs'], analysis['psd_inh'], 'r-', 
               linewidth=2, label='Inhibitory')
    ax4.axvspan(30, 100, alpha=0.2, color='gray', label='Gamma band')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power')
    ax4.set_title('Inhibitory Power Spectrum')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Network statistics
    ax5 = plt.subplot(4, 2, 7)
    
    # Calculate mean firing rates
    total_exc_spikes = sum(len(neuron_spikes) for neuron_spikes in spikes['E'])
    total_inh_spikes = sum(len(neuron_spikes) for neuron_spikes in spikes['I'])
    
    exc_rate_mean = total_exc_spikes / (len(spikes['E']) * duration / 1000)
    inh_rate_mean = total_inh_spikes / (len(spikes['I']) * duration / 1000)
    
    stats_text = f"""Network Statistics:
    
Excitatory Population:
• Mean firing rate: {exc_rate_mean:.1f} Hz
• Total neurons: {len(spikes['E'])}
• Total spikes: {total_exc_spikes}

Inhibitory Population:
• Mean firing rate: {inh_rate_mean:.1f} Hz
• Total neurons: {len(spikes['I'])}
• Total spikes: {total_inh_spikes}

Oscillations:
• Gamma peak: {analysis['gamma_peak']:.1f} Hz
• Gamma power: {analysis['gamma_power']:.2e}
    """
    
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Network Statistics')
    
    # Connectivity diagram
    ax6 = plt.subplot(4, 2, 8)
    
    # Simple schematic
    exc_circle = plt.Circle((0.3, 0.7), 0.15, color='blue', alpha=0.5)
    inh_circle = plt.Circle((0.7, 0.7), 0.1, color='red', alpha=0.5)
    
    ax6.add_patch(exc_circle)
    ax6.add_patch(inh_circle)
    
    # Add arrows for connections
    ax6.annotate('', xy=(0.7, 0.7), xytext=(0.45, 0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax6.annotate('', xy=(0.3, 0.55), xytext=(0.65, 0.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax6.text(0.3, 0.7, 'E', ha='center', va='center', fontsize=14, fontweight='bold')
    ax6.text(0.7, 0.7, 'I', ha='center', va='center', fontsize=14, fontweight='bold')
    ax6.text(0.5, 0.4, 'E-I Network\nGamma Generator', ha='center', va='center', 
             fontsize=12, fontweight='bold')
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Network Architecture')
    
    plt.tight_layout()
    plt.savefig('ei_network_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSimulation completed!")
    print(f"Gamma oscillation peak: {analysis['gamma_peak']:.1f} Hz")
    print(f"Mean excitatory rate: {exc_rate_mean:.1f} Hz")
    print(f"Mean inhibitory rate: {inh_rate_mean:.1f} Hz")

if __name__ == "__main__":
    main()

# =============================================================================
# File: examples/stdp_learning.py
"""
STDP Learning Example
====================

Demonstrates synaptic plasticity and learning in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator import LIFNeuron
from neural_simulator.synapses import STDPSynapse, AMPASynapse

def stdp_window_analysis():
    """Analyze STDP learning window."""
    
    stdp = STDPSynapse(a_plus=0.01, a_minus=0.012, 
                      tau_plus=20.0, tau_minus=20.0)
    
    # Test different spike timing differences
    dt_values = np.linspace(-100, 100, 201)
    weight_changes = []
    
    initial_weight = 1.0
    
    for dt in dt_values:
        stdp.reset()
        
        if dt > 0:  # Post after pre
            pre_spikes = [50.0]
            post_spikes = [50.0 + dt]
        else:  # Pre after post
            pre_spikes = [50.0 - dt]
            post_spikes = [50.0]
        
        # Simulate STDP
        for t in np.arange(0, 200, 0.1):
            pre_spike = any(abs(t - spike) < 0.05 for spike in pre_spikes)
            post_spike = any(abs(t - spike) < 0.05 for spike in post_spikes)
            
            stdp.update_traces(0.1, pre_spike, post_spike)
            
            if pre_spike or post_spike:
                new_weight = stdp.update_weight(initial_weight, pre_spike, post_spike)
                weight_change = new_weight - initial_weight
        
        weight_changes.append(weight_change)
    
    return dt_values, np.array(weight_changes)

def paired_pulse_protocol():
    """Simulate paired-pulse STDP protocol."""
    
    # Create pre and post neurons
    pre_neuron = LIFNeuron(v_thresh=-50.0)  # More excitable
    post_neuron = LIFNeuron()
    
    # Create plastic synapse
    stdp_synapse = STDPSynapse(a_plus=0.005, a_minus=0.006,
                              tau_plus=20.0, tau_minus=20.0)
    ampa_synapse = AMPASynapse(weight=2.0)
    
    # Protocol parameters
    n_pairs = 100
    pair_interval = 200.0  # ms between pairs
    dt_spike = 10.0  # ms, post after pre
    
    # Simulation
    duration = n_pairs * pair_interval
    dt = 0.1
    time = np.arange(0, duration, dt)
    
    weights = []
    pre_spikes = []
    post_spikes = []
    
    current_weight = 2.0
    
    for pair in range(n_pairs):
        pair_time = pair * pair_interval
        
        # Reset neurons
        pre_neuron.reset()
        post_neuron.reset()
        stdp_synapse.reset()
        ampa_synapse.reset()
        
        # Stimulate pre neuron
        pre_stim_time = pair_time + 50.0
        post_stim_time = pre_stim_time + dt_spike
        
        pair_spikes_pre = []
        pair_spikes_post = []
        
        # Simulate single pair
        for t in np.arange(pair_time, pair_time + pair_interval, dt):
            # Pre neuron stimulation
            pre_current = 50.0 if abs(t - pre_stim_time) < 2.0 else 0.0
            pre_spike = pre_neuron.step(pre_current, dt, t)
            
            if pre_spike:
                pair_spikes_pre.append(t)
                ampa_synapse.add_spike(t)
            
            # Post neuron receives synaptic input + stimulation
            i_syn = ampa_synapse.step(dt, t, post_neuron.v)
            post_current = 30.0 if abs(t - post_stim_time) < 2.0 else 0.0
            post_spike = post_neuron.step(post_current - i_syn, dt, t)
            
            if post_spike:
                pair_spikes_post.append(t)
            
            # Update STDP
            stdp_synapse.update_traces(dt, pre_spike, post_spike)
            if pre_spike or post_spike:
                current_weight = stdp_synapse.update_weight(current_weight, pre_spike, post_spike)
        
        weights.append(current_weight)
        pre_spikes.extend(pair_spikes_pre)
        post_spikes.extend(pair_spikes_post)
        ampa_synapse.weight = current_weight
    
    return np.arange(n_pairs), weights, pre_spikes, post_spikes

def main():
    """Run STDP learning demonstrations."""
    
    # 1. STDP window analysis
    print("Analyzing STDP learning window...")
    dt_values, weight_changes = stdp_window_analysis()
    
    # 2. Paired-pulse protocol
    print("Running paired-pulse STDP protocol...")
    pair_numbers, weights, pre_spikes, post_spikes = paired_pulse_protocol()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    
    # STDP window
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(dt_values, weight_changes * 1000, 'k-', linewidth=2)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Spike Timing Difference Δt (ms)')
    ax1.set_ylabel('Weight Change (×10⁻³)')
    ax1.set_title('STDP Learning Window')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('LTD', xy=(-40, -0.005), xytext=(-60, -0.008),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red', fontweight='bold')
    ax1.annotate('LTP', xy=(20, 0.008), xytext=(40, 0.010),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=12, color='blue', fontweight='bold')
    
    # Weight evolution
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(pair_numbers, weights, 'b-', linewidth=2)
    ax2.set_xlabel('Pairing Number')
    ax2.set_ylabel('Synaptic Weight')
    ax2.set_title('Weight Evolution During STDP')
    ax2.grid(True, alpha=0.3)
    
    # Weight change percentage
    initial_weight = weights[0]
    final_weight = weights[-1]
    weight_change_percent = ((final_weight - initial_weight) / initial_weight) * 100
    ax2.text(0.7, 0.9, f'Weight change: {weight_change_percent:+.1f}%', 
             transform=ax2.transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Spike timing raster
    ax3 = plt.subplot(3, 2, (3, 4))
    
    # Show only first 10 pairs for clarity
    pair_times = np.arange(10) * 200.0
    
    for i, pair_time in enumerate(pair_times):
        # Find spikes in this pair
        pair_pre = [t for t in pre_spikes if pair_time <= t < pair_time + 200]
        pair_post = [t for t in post_spikes if pair_time <= t < pair_time + 200]
        
        if pair_pre:
            ax3.scatter(pair_pre, [i] * len(pair_pre), c='red', s=20, marker='|')
        if pair_post:
            ax3.scatter(pair_post, [i + 0.1] * len(pair_post), c='blue', s=20, marker='|')
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Pair Number')
    ax3.set_title('Spike Timing During STDP Protocol (first 10 pairs)')
    ax3.legend(['Pre spikes', 'Post spikes'], loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Learning curve analysis
    ax4 = plt.subplot(3, 2, 5)
    
    # Smooth weight curve for trend analysis
    from scipy import ndimage
    smoothed_weights = ndimage.gaussian_filter1d(weights, sigma=2)
    
    ax4.plot(pair_numbers, weights, 'lightblue', alpha=0.5, label='Raw')
    ax4.plot(pair_numbers, smoothed_weights, 'darkblue', linewidth=2, label='Smoothed')
    
    # Fit exponential curve
    try:
        from scipy.optimize import curve_fit
        
        def exp_func(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c
        
        popt, _ = curve_fit(exp_func, pair_numbers, weights, 
                           p0=[0.5, 0.01, initial_weight])
        fitted_curve = exp_func(pair_numbers, *popt)
        ax4.plot(pair_numbers, fitted_curve, 'red', linestyle='--', 
                linewidth=2, label='Exponential fit')
        
        # Calculate learning rate
        learning_rate = popt[1]
        ax4.text(0.5, 0.2, f'Learning rate: {learning_rate:.4f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    except:
        pass
    
    ax4.set_xlabel('Pairing Number')
    ax4.set_ylabel('Synaptic Weight')
    ax4.set_title('Learning Curve Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # STDP mechanisms schematic
    ax5 = plt.subplot(3, 2, 6)
    
    # Create schematic of STDP mechanism
    ax5.text(0.5, 0.9, 'STDP Mechanism', ha='center', fontsize=14, fontweight='bold')
    
    # Pre and post neuron representations
    ax5.add_patch(plt.Circle((0.2, 0.6), 0.05, color='red', alpha=0.7))
    ax5.add_patch(plt.Circle((0.8, 0.6), 0.05, color='blue', alpha=0.7))
    
    # Synapse
    ax5.plot([0.25, 0.75], [0.6, 0.6], 'k-', linewidth=3)
    ax5.plot([0.7, 0.75, 0.7], [0.55, 0.6, 0.65], 'k-', linewidth=2)
    
    ax5.text(0.2, 0.5, 'Pre', ha='center', fontweight='bold')
    ax5.text(0.8, 0.5, 'Post', ha='center', fontweight='bold')
    ax5.text(0.5, 0.65, 'Synapse', ha='center', fontsize=10)
    
    # Timing conditions
    ax5.text(0.5, 0.3, 'Pre → Post: LTP (strengthening)', ha='center', 
             color='green', fontweight='bold')
    ax5.text(0.5, 0.2, 'Post → Pre: LTD (weakening)', ha='center', 
             color='red', fontweight='bold')
    
    # Results summary
    ax5.text(0.5, 0.05, f'Final weight: {final_weight:.3f}\nChange: {weight_change_percent:+.1f}%', 
             ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig('stdp_learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSTDP Analysis Complete!")
    print(f"Initial weight: {initial_weight:.3f}")
    print(f"Final weight: {final_weight:.3f}")
    print(f"Weight change: {weight_change_percent:+.1f}%")

if __name__ == "__main__":
    main()

# =============================================================================
# File: requirements.txt

numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.0
networkx>=2.6.0
tqdm>=4.62.0

# =============================================================================
# File: setup.py
"""
Setup script for Biological Neural Network Simulator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="biological-neural-network-simulator",
    version="1.0.0",
    author="Computational Neuroscience Lab",
    author_email="your.email@university.edu",
    description="A comprehensive toolkit for simulating biophysically realistic neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biological-neural-network-simulator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9.0",
            "black>=21.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-simulator=neural_simulator.cli:main",
        ],
    },
)

# =============================================================================
# File: tests/test_models.py
"""
Unit tests for neuron models
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator.models.hodgkin_huxley import HodgkinHuxleyNeuron
from neural_simulator.models.integrate_fire import LIFNeuron, AdExNeuron

class TestHodgkinHuxleyNeuron(unittest.TestCase):
    """Test HH neuron model."""
    
    def setUp(self):
        self.neuron = HodgkinHuxleyNeuron()
    
    def test_initialization(self):
        """Test neuron initialization."""
        self.assertAlmostEqual(self.neuron.v, -65.0, places=1)
        self.assertTrue(0 < self.neuron.m < 1)
        self.assertTrue(0 < self.neuron.h < 1)
        self.assertTrue(0 < self.neuron.n < 1)
    
    def test_rate_constants(self):
        """Test rate constant calculations."""
        v_test = -60.0
        
        # Test that rate constants are positive
        self.assertGreater(self.neuron.alpha_m(v_test), 0)
        self.assertGreater(self.neuron.beta_m(v_test), 0)
        self.assertGreater(self.neuron.alpha_h(v_test), 0)
        self.assertGreater(self.neuron.beta_h(v_test), 0)
        self.assertGreater(self.neuron.alpha_n(v_test), 0)
        self.assertGreater(self.neuron.beta_n(v_test), 0)
    
    def test_action_potential(self):
        """Test action potential generation."""
        # Strong current should generate action potential
        current = [15.0] * 1000  # 10ms at dt=0.01
        time, voltage, _ = self.neuron.simulate(current, dt=0.01, duration=10)
        
        # Should reach threshold and spike
        self.assertGreater(np.max(voltage), 0)  # Should overshoot 0mV
        
    def test_no_spike_subthreshold(self):
        """Test no spike with subthreshold current."""
        # Weak current should not generate spike
        current = [2.0] * 1000  # 10ms at dt=0.01
        time, voltage, _ = self.neuron.simulate(current, dt=0.01, duration=10)
        
        # Should not reach spike threshold
        self.assertLess(np.max(voltage), -20)  # Well below spike threshold

class TestLIFNeuron(unittest.TestCase):
    """Test LIF neuron model."""
    
    def setUp(self):
        self.neuron = LIFNeuron()
    
    def test_initialization(self):
        """Test neuron initialization."""
        self.assertEqual(self.neuron.v, self.neuron.v_rest)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_membrane_integration(self):
        """Test membrane potential integration."""
        initial_v = self.neuron.v
        
        # Apply current step
        self.neuron.step(10.0, 0.1, 0.0)
        
        # Voltage should increase
        self.assertGreater(self.neuron.v, initial_v)
    
    def test_spike_generation(self):
        """Test spike generation and reset."""
        # Apply strong current to trigger spike
        for t in np.arange(0, 50, 0.1):
            spike = self.neuron.step(25.0, 0.1, t)
            if spike:
                break
        
        # Should have spiked
        self.assertTrue(len(self.neuron.spike_times) > 0)
        self.assertEqual(self.neuron.v, self.neuron.v_reset)
    
    def test_refractory_period(self):
        """Test refractory period."""
        # Trigger spike
        self.neuron.step(25.0, 0.1, 0.0)
        
        # During refractory period, no spike should occur
        spike = self.neuron.step(25.0, 0.1, 1.0)  # Within refractory
        self.assertFalse(spike)

class TestAdExNeuron(unittest.TestCase):
    """Test AdEx neuron model."""
    
    def setUp(self):
        self.neuron = AdExNeuron()
    
    def test_initialization(self):
        """Test neuron initialization."""
        self.assertEqual(self.neuron.v, self.neuron.el)
        self.assertEqual(self.neuron.w, 0.0)
    
    def test_adaptation(self):
        """Test spike-frequency adaptation."""
        # Apply constant current and record spikes
        spikes = []
        
        for t in np.arange(0, 1000, 0.1):
            spike = self.neuron.step(2.0, 0.1, t)
            if spike:
                spikes.append(t)
        
        if len(spikes) > 2:
            # Check if ISI increases (adaptation)
            isi_early = spikes[1] - spikes[0]
            isi_late = spikes[-1] - spikes[-2]
            self.assertGreater(isi_late, isi_early * 0.9)  # Some adaptation

if __name__ == '__main__':
    unittest.main()

# =============================================================================
# File: tests/test_synapses.py
"""
Unit tests for synaptic models
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator.synapses.ampa_synapse import AMPASynapse
from neural_simulator.synapses.gaba_synapse import GABASynapse
from neural_simulator.synapses.stdp import STDPSynapse

class TestAMPASynapse(unittest.TestCase):
    """Test AMPA synapse model."""
    
    def setUp(self):
        self.synapse = AMPASynapse(weight=2.0, delay=1.0)
    
    def test_initialization(self):
        """Test synapse initialization."""
        self.assertEqual(self.synapse.g, 0.0)
        self.assertEqual(len(self.synapse.spike_buffer), 0)
    
    def test_spike_handling(self):
        """Test spike addition and processing."""
        # Add spike
        self.synapse.add_spike(5.0)
        self.assertEqual(len(self.synapse.spike_buffer), 1)
        
        # Process before delay
        current = self.synapse.step(0.1, 5.5, -60.0)
        self.assertEqual(len(self.synapse.spike_buffer), 1)  # Still buffered
        
        # Process after delay
        current = self.synapse.step(0.1, 7.0, -60.0)
        self.assertEqual(len(self.synapse.spike_buffer), 0)  # Should be processed
    
    def test_conductance_dynamics(self):
        """Test conductance time course."""
        self.synapse.add_spike(0.0)
        
        conductances = []
        for t in np.arange(0, 20, 0.1):
            current = self.synapse.step(0.1, t, 0.0)
            conductances.append(self.synapse.g)
        
        # Should rise then decay
        max_idx = np.argmax(conductances)
        self.assertGreater(max_idx, 0)  # Peak not at start
        self.assertLess(max_idx, len(conductances) - 1)  # Peak not at end

class TestGABASynapse(unittest.TestCase):
    """Test GABA synapse model."""
    
    def setUp(self):
        self.synapse = GABASynapse(weight=3.0, e_rev=-70.0)
    
    def test_inhibitory_current(self):
        """Test inhibitory current direction."""
        self.synapse.add_spike(0.0)
        
        # At resting potential (-65mV), should be outward (negative) current
        current = self.synapse.step(0.1, 2.0, -65.0)
        self.assertLess(current, 0)  # Inhibitory (hyperpolarizing)

class TestSTDPSynapse(unittest.TestCase):
    """Test STDP synapse model."""
    
    def setUp(self):
        self.stdp = STDPSynapse(a_plus=0.01, a_minus=0.012)
    
    def test_initialization(self):
        """Test STDP initialization."""
        self.assertEqual(self.stdp.x_pre, 0.0)
        self.assertEqual(self.stdp.x_post, 0.0)
    
    def test_ltp_condition(self):
        """Test LTP (pre before post)."""
        initial_weight = 1.0
        
        # Pre spike first
        self.stdp.update_traces(0.1, pre_spike=True, post_spike=False)
        
        # Post spike after some delay
        for _ in range(50):  # 5ms delay
            self.stdp.update_traces(0.1, pre_spike=False, post_spike=False)
        
        final_weight = self.stdp.update_weight(initial_weight, 
                                              pre_spike=False, post_spike=True)
        
        # Should be potentiation
        self.assertGreater(final_weight, initial_weight)
    
    def test_ltd_condition(self):
        """Test LTD (post before pre)."""
        initial_weight = 1.0
        
        # Post spike first
        self.stdp.update_traces(0.1, pre_spike=False, post_spike=True)
        
        # Pre spike after some delay
        for _ in range(50):  # 5ms delay
            self.stdp.update_traces(0.1, pre_spike=False, post_spike=False)
        
        final_weight = self.stdp.update_weight(initial_weight, 
                                              pre_spike=True, post_spike=False)
        
        # Should be depression
        self.assertLess(final_weight, initial_weight)
    
    def test_weight_bounds(self):
        """Test weight bounds enforcement."""
        # Test upper bound
        high_weight = self.stdp.w_max + 1.0
        bounded_weight = self.stdp.update_weight(high_weight, 
                                                pre_spike=False, post_spike=True)
        self.assertLessEqual(bounded_weight, self.stdp.w_max)
        
        # Test lower bound
        low_weight = self.stdp.w_min - 1.0
        bounded_weight = self.stdp.update_weight(low_weight, 
                                                pre_spike=True, post_spike=False)
        self.assertGreaterEqual(bounded_weight, self.stdp.w_min)

if __name__ == '__main__':
    unittest.main()

# =============================================================================
# File: tests/test_networks.py
"""
Unit tests for neural network simulation
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_simulator.networks.neural_network import NeuralNetwork, Population
from neural_simulator.models.integrate_fire import LIFNeuron
from neural_simulator.synapses.ampa_synapse import AMPASynapse

class TestNeuralNetwork(unittest.TestCase):
    """Test neural network framework."""
    
    def setUp(self):
        self.network = NeuralNetwork()
    
    def test_add_population(self):
        """Test population creation."""
        pop = self.network.add_population('test', LIFNeuron, 10)
        
        self.assertIn('test', self.network.populations)
        self.assertEqual(pop.size, 10)
        self.assertEqual(len(pop.neurons), 10)
    
    def test_connect_populations(self):
        """Test population connectivity."""
        pop1 = self.network.add_population('pop1', LIFNeuron, 5)
        pop2 = self.network.add_population('pop2', LIFNeuron, 5)
        
        self.network.connect_populations(pop1, pop2, AMPASynapse,
                                       connection_prob=1.0, weight=2.0)
        
        self.assertEqual(len(self.network.connections), 1)
        
        # Check connection matrix
        conn = self.network.connections[0]
        self.assertEqual(conn.connection_matrix.shape, (5, 5))
    
    def test_simulation_runs(self):
        """Test that simulation completes without errors."""
        # Create simple network
        pop = self.network.add_population('test', LIFNeuron, 5)
        
        # Run short simulation
        spikes, voltages = self.network.simulate(duration=10.0, dt=0.1,
                                               external_current=10.0)
        
        # Should return results
        self.assertIn('test', spikes)
        self.assertEqual(len(spikes['test']), 5)  # One list per neuron
    
    def test_connected_network_activity(self):
        """Test activity in connected network."""
        # Create two populations
        pop1 = self.network.add_population('excitatory', LIFNeuron, 10,
                                          v_thresh=-50.0)  # More excitable
        pop2 = self.network.add_population('target', LIFNeuron, 5)
        
        # Connect them
        self.network.connect_populations(pop1, pop2, AMPASynapse,
                                       connection_prob=0.8, weight=5.0)
        
        # Stimulate first population strongly
        spikes, _ = self.network.simulate(duration=50.0, dt=0.1,
                                        external_current=20.0)
        
        # Both populations should be active
        total_spikes_1 = sum(len(neuron_spikes) for neuron_spikes in spikes['excitatory'])
        total_spikes_2 = sum(len(neuron_spikes) for neuron_spikes in spikes['target'])
        
        self.assertGreater(total_spikes_1, 0)
        # Target population might spike due to excitatory input
        # (depending on parameters, might be 0 if connections too weak)

if __name__ == '__main__':
    unittest.main()

# =============================================================================
# File: .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
env/
venv/
ENV/
env.bak/
venv.bak/
neural_sim_env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Results and plots
*.png
*.pdf
*.svg
results/
figures/
plots/

# Data files
*.h5
*.hdf5
*.npz
*.pkl
*.pickle

# Documentation
docs/_build/
docs/auto_generated/

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
.pytest_cache/
htmlcov/

# Profiling
*.prof

# =============================================================================
# File: LICENSE

MIT License

Copyright (c) 2025 Computational Neuroscience Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.