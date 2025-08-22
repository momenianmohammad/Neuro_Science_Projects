#!/usr/bin/env python3
"""
Example Usage of Sleep Analysis Tool
===================================

This script demonstrates how to use the Sleep and Dream Pattern Analysis Tool
for analyzing EEG data and generating comprehensive sleep reports.

Author: [Your Name]
Institution: [Your University]
"""

from sleep_analyzer import SleepAnalyzer
import numpy as np
from datetime import datetime
import os

def example_with_real_edf_file():
    """
    Example of analyzing a real EDF file.
    Replace 'path/to/your/eeg_file.edf' with actual EDF file path.
    """
    print("📄 Example 1: Analyzing Real EDF File")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = SleepAnalyzer(sampling_rate=256, epoch_length=30)
    
    # Load real EEG data (uncomment and modify path)
    # eeg_data = analyzer.load_eeg_data('path/to/your/eeg_file.edf')
    
    # For demonstration, we'll use synthetic data
    eeg_data = create_sample_data(analyzer)
    
    # Patient information
    patient_info = {
        'name': 'John Doe',
        'age': '45',
        'date': '2024-01-15',
        'study_id': 'PSG_001'
    }
    
    # Run complete analysis
    results = analyzer.analyze_sleep(eeg_data, patient_info, save_plots=True)
    
    # Access specific results
    print(f"Sleep Quality Score: {results['quality_score']:.1f}/100")
    print(f"Sleep Efficiency: {results['metrics']['sleep_efficiency']:.1f}%")
    
    return results

def example_batch_processing():
    """
    Example of processing multiple sleep studies.
    """
    print("\n📊 Example 2: Batch Processing Multiple Studies")
    print("=" * 50)
    
    analyzer = SleepAnalyzer()
    
    # List of study files (in real use, these would be actual file paths)
    study_files = [
        {'file': 'patient_001.edf', 'name': 'Patient 001', 'age': '30'},
        {'file': 'patient_002.edf', 'name': 'Patient 002', 'age': '45'},
        {'file': 'patient_003.edf', 'name': 'Patient 003', 'age': '60'}
    ]
    
    batch_results = []
    
    for i, study in enumerate(study_files):
        print(f"\nProcessing study {i+1}/{len(study_files)}: {study['name']}")
        
        # Create synthetic data for each patient (replace with actual file loading)
        eeg_data = create_sample_data(analyzer, variation=i)
        
        patient_info = {
            'name': study['name'],
            'age': study['age'],
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Analyze sleep
        results = analyzer.analyze_sleep(eeg_data, patient_info, save_plots=False)
        
        # Store results
        batch_results.append({
            'patient': study['name'],
            'quality_score': results['quality_score'],
            'sleep_efficiency': results['metrics']['sleep_efficiency'],
            'total_sleep_time': results['metrics']['total_sleep_time']
        })
        
        # Save individual report
        report_filename = f"report_{study['name'].replace(' ', '_').lower()}.txt"
        with open(report_filename, 'w') as f:
            f.write(results['report'])
        print(f"✓ Report saved: {report_filename}")
    
    # Summary statistics
    print("\n📈 Batch Processing Summary:")
    print("-" * 30)
    for result in batch_results:
        print(f"{result['patient']:12} | Quality: {result['quality_score']:5.1f} | "
              f"Efficiency: {result['sleep_efficiency']:5.1f}% | "
              f"TST: {result['total_sleep_time']/60:4.1f}h")
    
    return batch_results

def example_custom_analysis():
    """
    Example of customized analysis with specific parameters.
    """
    print("\n🔧 Example 3: Custom Analysis Configuration")
    print("=" * 45)
    
    # Initialize with custom parameters
    analyzer = SleepAnalyzer(sampling_rate=200, epoch_length=20)
    
    # Modify frequency bands for specific research
    analyzer.frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'sigma': (11, 15),  # Sleep spindles
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # Create sample data
    eeg_data = create_sample_data(analyzer)
    
    # Custom patient info
    patient_info = {
        'name': 'Research Subject 001',
        'age': '25',
        'condition': 'Healthy Control',
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Run analysis
    results = analyzer.analyze_sleep(eeg_data, patient_info)
    
    # Custom post-processing
    sleep_stages = results['sleep_stages']
    
    # Calculate additional metrics
    sleep_spindle_density = calculate_sleep_spindle_density(sleep_stages)
    print(f"Custom Metric - Sleep Spindle Density: {sleep_spindle_density:.2f}/hour")
    
    return results

def calculate_sleep_spindle_density(sleep_stages):
    """
    Example of custom metric calculation.
    """
    # This is a simplified example - in reality, you'd analyze the actual EEG
    n2_epochs = np.sum(np.array(sleep_stages['predictions']) == 2)  # N2 stage
    total_hours = len(sleep_stages['predictions']) * 30 / 3600  # Convert epochs to hours
    
    # Simulate spindle detection (replace with actual spindle detection algorithm)
    simulated_spindles = n2_epochs * 0.8  # Approximate spindles per N2 epoch
    spindle_density = simulated_spindles / total_hours if total_hours > 0 else 0
    
    return spindle_density

def create_sample_data(analyzer, variation=0):
    """
    Create sample EEG data for demonstration purposes.
    """
    duration_hours = 7 + variation * 0.5  # Vary duration slightly
    duration_samples = int(duration_hours * 3600 * analyzer.sampling_rate)
    
    # Generate synthetic EEG with realistic sleep patterns
    t = np.linspace(0, duration_hours * 3600, duration_samples)
    
    # Two-channel EEG
    channel1 = np.zeros(duration_samples)
    channel2 = np.zeros(duration_samples)
    
    for i, time_point in enumerate(t):
        hour = time_point / 3600
        noise_factor = 1 + variation * 0.2  # Add variation between patients
        
        if hour < 0.3:  # Sleep onset
            channel1[i] = np.random.randn() * 0.4 * noise_factor + 0.5 * np.sin(2 * np.pi * 8 * time_point)
            channel2[i] = np.random.randn() * 0.4 * noise_factor + 0.3 * np.sin(2 * np.pi * 10 * time_point)
        elif hour < 1.5:  # Light sleep
            channel1[i] = np.random.randn() * 0.3 * noise_factor + np.sin(2 * np.pi * 6 * time_point)
            channel2[i] = np.random.randn() * 0.3 * noise_factor + 0.8 * np.sin(2 * np.pi * 7 * time_point)
        elif hour < 3.5:  # Deep sleep
            channel1[i] = np.random.randn() * 0.2 * noise_factor + 2 * np.sin(2 * np.pi * 1.2 * time_point)
            channel2[i] = np.random.randn() * 0.2 * noise_factor + 1.8 * np.sin(2 * np.pi * 0.9 * time_point)
        elif hour < 5.5:  # REM periods
            channel1[i] = np.random.randn() * 0.35 * noise_factor + 0.7 * np.sin(2 * np.pi * 9 * time_point)
            channel2[i] = np.random.randn() * 0.35 * noise_factor + 0.5 * np.sin(2 * np.pi * 11 * time_point)
        else:  # Morning light sleep and wake
            channel1[i] = np.random.randn() * 0.4 * noise_factor + 0.8 * np.sin(2 * np.pi * 12 * time_point)
            channel2[i] = np.random.randn() * 0.4 * noise_factor + np.sin(2 * np.pi * 16 * time_point)
    
    return {
        'data': np.array([channel1, channel2]),
        'channel_names': ['C3-A2', 'C4-A1'],
        'sampling_rate': analyzer.sampling_rate,
        'duration': duration_hours * 3600
    }

def main():
    """
    Main function to run all examples.
    """
    print("🌙 Sleep Analysis Tool - Usage Examples")
    print("=" * 45)
    print("This script demonstrates various ways to use the Sleep Analysis Tool")
    print()
    
    # Create output directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Run examples
        results1 = example_with_real_edf_file()
        results2 = example_batch_processing()
        results3 = example_custom_analysis()
        
        print("\n🎉 All examples completed successfully!")
        print("\nGenerated files:")
        print("- sleep_analysis_plots.png")
        print("- sleep_analysis_report.txt")
        print("- Individual patient reports")
        
        # Summary statistics
        print(f"\nExample Quality Scores:")
        print(f"- Single Analysis: {results1['quality_score']:.1f}/100")
        print(f"- Batch Average: {np.mean([r['quality_score'] for r in results2]):.1f}/100")
        print(f"- Custom Analysis: {results3['quality_score']:.1f}/100")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()