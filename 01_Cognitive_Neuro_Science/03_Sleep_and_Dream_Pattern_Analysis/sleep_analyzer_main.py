"""
Sleep and Dream Pattern Analysis Tool
=====================================

A comprehensive system for analyzing sleep stages from EEG and PSG data,
predicting sleep quality, and providing insights into sleep patterns.

Author: [Your Name]
Institution: [Your University]
Email: [your.email@university.edu]
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mne
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SleepAnalyzer:
    """
    Main class for sleep pattern analysis and stage classification.
    """
    
    def __init__(self, sampling_rate=256, epoch_length=30):
        """
        Initialize the Sleep Analyzer.
        
        Parameters:
        -----------
        sampling_rate : int
            EEG sampling rate in Hz (default: 256)
        epoch_length : int
            Length of each sleep epoch in seconds (default: 30)
        """
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.samples_per_epoch = sampling_rate * epoch_length
        
        # Frequency bands for sleep stage analysis
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Sleep stage labels
        self.sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
        
        # Initialize models
        self.classifier = None
        self.scaler = StandardScaler()
        
    def load_eeg_data(self, file_path):
        """
        Load EEG data from file (supports EDF, BDF, etc.)
        
        Parameters:
        -----------
        file_path : str
            Path to the EEG data file
            
        Returns:
        --------
        dict : Loaded EEG data with metadata
        """
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Basic preprocessing
            raw.filter(0.3, 100, fir_design='firwin')
            raw.notch_filter(50, fir_design='firwin')  # Remove power line noise
            
            data = {
                'raw': raw,
                'data': raw.get_data(),
                'channel_names': raw.ch_names,
                'sampling_rate': raw.info['sfreq'],
                'duration': raw.times[-1]
            }
            
            print(f"✓ Loaded EEG data: {len(data['channel_names'])} channels, "
                  f"{data['duration']:.1f} seconds")
            
            return data
            
        except Exception as e:
            print(f"✗ Error loading EEG data: {str(e)}")
            return None
    
    def extract_features(self, epoch_data):
        """
        Extract sleep-related features from EEG epoch.
        
        Parameters:
        -----------
        epoch_data : np.array
            EEG data for one epoch (channels x samples)
            
        Returns:
        --------
        dict : Extracted features
        """
        features = {}
        
        for ch_idx, channel_data in enumerate(epoch_data):
            ch_name = f'ch_{ch_idx}'
            
            # Power spectral density features
            freqs, psd = signal.welch(channel_data, self.sampling_rate, 
                                    nperseg=min(len(channel_data), 256))
            
            # Band power features
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features[f'{ch_name}_{band_name}_power'] = band_power
                
                # Relative power
                total_power = np.trapz(psd, freqs)
                features[f'{ch_name}_{band_name}_rel_power'] = band_power / total_power
            
            # Statistical features
            features[f'{ch_name}_mean'] = np.mean(channel_data)
            features[f'{ch_name}_std'] = np.std(channel_data)
            features[f'{ch_name}_skewness'] = signal.stats.skew(channel_data)
            features[f'{ch_name}_kurtosis'] = signal.stats.kurtosis(channel_data)
            
            # Entropy features
            hist, _ = np.histogram(channel_data, bins=50)
            hist = hist / np.sum(hist)
            features[f'{ch_name}_entropy'] = entropy(hist + 1e-10)
            
            # Hjorth parameters
            features[f'{ch_name}_activity'] = np.var(channel_data)
            diff1 = np.diff(channel_data)
            diff2 = np.diff(diff1)
            features[f'{ch_name}_mobility'] = np.sqrt(np.var(diff1) / np.var(channel_data))
            features[f'{ch_name}_complexity'] = (np.sqrt(np.var(diff2) / np.var(diff1)) / 
                                               features[f'{ch_name}_mobility'])
        
        # Cross-channel coherence (simplified for 2-channel case)
        if len(epoch_data) >= 2:
            freqs, coherence = signal.coherence(epoch_data[0], epoch_data[1], 
                                              self.sampling_rate)
            features['coherence_mean'] = np.mean(coherence)
            features['coherence_max'] = np.max(coherence)
        
        return features
    
    def create_synthetic_training_data(self, n_samples=1000):
        """
        Create synthetic training data for sleep stage classification.
        In a real implementation, this would be replaced with labeled sleep data.
        
        Parameters:
        -----------
        n_samples : int
            Number of training samples to generate
            
        Returns:
        --------
        tuple : (X, y) training features and labels
        """
        print("Creating synthetic training data...")
        
        X = []
        y = []
        
        for stage_idx, stage in enumerate(self.sleep_stages):
            for _ in range(n_samples // len(self.sleep_stages)):
                # Generate synthetic EEG epoch based on sleep stage characteristics
                if stage == 'Wake':
                    # High beta, low delta
                    epoch = (np.random.randn(2, self.samples_per_epoch) * 0.3 + 
                           np.sin(2 * np.pi * 15 * np.linspace(0, self.epoch_length, self.samples_per_epoch)))
                elif stage == 'N1':
                    # Mixed frequencies, theta prominence
                    epoch = (np.random.randn(2, self.samples_per_epoch) * 0.5 + 
                           np.sin(2 * np.pi * 6 * np.linspace(0, self.epoch_length, self.samples_per_epoch)))
                elif stage == 'N2':
                    # Sleep spindles (12-14 Hz) and K-complexes
                    epoch = (np.random.randn(2, self.samples_per_epoch) * 0.4 + 
                           0.5 * np.sin(2 * np.pi * 13 * np.linspace(0, self.epoch_length, self.samples_per_epoch)))
                elif stage == 'N3':
                    # High delta waves
                    epoch = (np.random.randn(2, self.samples_per_epoch) * 0.2 + 
                           2 * np.sin(2 * np.pi * 2 * np.linspace(0, self.epoch_length, self.samples_per_epoch)))
                else:  # REM
                    # Similar to wake but with different patterns
                    epoch = (np.random.randn(2, self.samples_per_epoch) * 0.4 + 
                           np.sin(2 * np.pi * 8 * np.linspace(0, self.epoch_length, self.samples_per_epoch)))
                
                features = self.extract_features(epoch)
                X.append(list(features.values()))
                y.append(stage_idx)
        
        return np.array(X), np.array(y)
    
    def train_classifier(self, X=None, y=None):
        """
        Train the sleep stage classifier.
        
        Parameters:
        -----------
        X : np.array, optional
            Training features (if None, synthetic data is used)
        y : np.array, optional
            Training labels
        """
        if X is None or y is None:
            X, y = self.create_synthetic_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"✓ Classifier trained. Accuracy: {accuracy:.3f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.sleep_stages))
        
        return accuracy
    
    def classify_sleep_stages(self, eeg_data):
        """
        Classify sleep stages for entire EEG recording.
        
        Parameters:
        -----------
        eeg_data : dict
            EEG data from load_eeg_data()
            
        Returns:
        --------
        dict : Sleep stage classification results
        """
        if self.classifier is None:
            print("Training classifier...")
            self.train_classifier()
        
        data = eeg_data['data']
        n_channels, n_samples = data.shape
        n_epochs = n_samples // self.samples_per_epoch
        
        print(f"Classifying {n_epochs} epochs...")
        
        predictions = []
        probabilities = []
        epoch_times = []
        
        for epoch_idx in range(n_epochs):
            start_idx = epoch_idx * self.samples_per_epoch
            end_idx = start_idx + self.samples_per_epoch
            
            epoch_data = data[:, start_idx:end_idx]
            
            # Extract features
            features = self.extract_features(epoch_data)
            X_epoch = np.array(list(features.values())).reshape(1, -1)
            
            # Predict
            X_scaled = self.scaler.transform(X_epoch)
            prediction = self.classifier.predict(X_scaled)[0]
            probability = self.classifier.predict_proba(X_scaled)[0]
            
            predictions.append(prediction)
            probabilities.append(probability)
            epoch_times.append(epoch_idx * self.epoch_length / 60)  # Convert to minutes
        
        results = {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'epoch_times': np.array(epoch_times),
            'stage_names': [self.sleep_stages[pred] for pred in predictions]
        }
        
        return results
    
    def calculate_sleep_metrics(self, sleep_stages):
        """
        Calculate comprehensive sleep quality metrics.
        
        Parameters:
        -----------
        sleep_stages : dict
            Results from classify_sleep_stages()
            
        Returns:
        --------
        dict : Sleep metrics
        """
        predictions = sleep_stages['predictions']
        epoch_times = sleep_stages['epoch_times']
        
        # Basic counts
        stage_counts = {
            stage: np.sum(predictions == idx) 
            for idx, stage in enumerate(self.sleep_stages)
        }
        
        total_epochs = len(predictions)
        total_time = total_epochs * self.epoch_length / 60  # minutes
        
        # Sleep efficiency
        sleep_epochs = total_epochs - stage_counts['Wake']
        sleep_efficiency = (sleep_epochs / total_epochs) * 100
        
        # Total sleep time
        total_sleep_time = sleep_epochs * self.epoch_length / 60
        
        # Sleep onset latency (time to first sleep stage)
        sleep_onset_idx = np.where(predictions != 0)[0]
        sleep_onset_latency = sleep_onset_idx[0] * self.epoch_length / 60 if len(sleep_onset_idx) > 0 else 0
        
        # REM latency (time to first REM)
        rem_idx = np.where(predictions == 4)[0]
        rem_latency = rem_idx[0] * self.epoch_length / 60 if len(rem_idx) > 0 else 0
        
        # Stage percentages
        stage_percentages = {
            stage: (count / sleep_epochs * 100) if sleep_epochs > 0 else 0
            for stage, count in stage_counts.items()
            if stage != 'Wake'
        }
        
        # Sleep fragmentation (number of stage transitions)
        transitions = np.sum(np.diff(predictions) != 0)
        fragmentation_index = transitions / total_sleep_time if total_sleep_time > 0 else 0
        
        # Deep sleep ratio
        deep_sleep_ratio = stage_percentages['N3']
        
        # Sleep quality score (0-100)
        quality_score = (
            sleep_efficiency * 0.3 +
            (100 - min(sleep_onset_latency * 2, 100)) * 0.2 +
            deep_sleep_ratio * 0.2 +
            stage_percentages['REM'] * 0.15 +
            (100 - min(fragmentation_index * 10, 100)) * 0.15
        )
        
        metrics = {
            'total_recording_time': total_time,
            'total_sleep_time': total_sleep_time,
            'sleep_efficiency': sleep_efficiency,
            'sleep_onset_latency': sleep_onset_latency,
            'rem_latency': rem_latency,
            'stage_counts': stage_counts,
            'stage_percentages': stage_percentages,
            'transitions': transitions,
            'fragmentation_index': fragmentation_index,
            'quality_score': quality_score
        }
        
        return metrics
    
    def visualize_sleep_patterns(self, sleep_stages, metrics, save_path=None):
        """
        Create comprehensive sleep pattern visualizations.
        
        Parameters:
        -----------
        sleep_stages : dict
            Results from classify_sleep_stages()
        metrics : dict
            Results from calculate_sleep_metrics()
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sleep Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hypnogram
        ax1 = axes[0, 0]
        epoch_times = sleep_stages['epoch_times']
        predictions = sleep_stages['predictions']
        
        ax1.plot(epoch_times, predictions, 'b-', linewidth=1)
        ax1.fill_between(epoch_times, predictions, alpha=0.3)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Sleep Stage')
        ax1.set_title('Hypnogram')
        ax1.set_yticks(range(len(self.sleep_stages)))
        ax1.set_yticklabels(self.sleep_stages)
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. Stage Distribution Pie Chart
        ax2 = axes[0, 1]
        stage_counts = metrics['stage_counts']
        non_zero_stages = {k: v for k, v in stage_counts.items() if v > 0}
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        ax2.pie(non_zero_stages.values(), labels=non_zero_stages.keys(), 
                autopct='%1.1f%%', colors=colors[:len(non_zero_stages)])
        ax2.set_title('Sleep Stage Distribution')
        
        # 3. Sleep Metrics Bar Chart
        ax3 = axes[1, 0]
        metrics_to_plot = {
            'Sleep Efficiency (%)': metrics['sleep_efficiency'],
            'Deep Sleep (%)': metrics['stage_percentages']['N3'],
            'REM Sleep (%)': metrics['stage_percentages']['REM'],
            'Quality Score': metrics['quality_score']
        }
        
        bars = ax3.bar(metrics_to_plot.keys(), metrics_to_plot.values(), 
                      color=['#45B7D1', '#96CEB4', '#FECA57', '#FF6B6B'])
        ax3.set_title('Sleep Quality Metrics')
        ax3.set_ylabel('Percentage / Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_to_plot.values()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Sleep Architecture Timeline
        ax4 = axes[1, 1]
        
        # Create color-coded timeline
        colors_map = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#96CEB4', 4: '#FECA57'}
        for i, stage in enumerate(predictions):
            ax4.barh(0, self.epoch_length/60, left=epoch_times[i], 
                    color=colors_map[stage], height=0.8)
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_title('Sleep Architecture')
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_yticks([])
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors_map[i], 
                                       label=stage) for i, stage in enumerate(self.sleep_stages)]
        ax4.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Sleep pattern visualization saved to {save_path}")
        
        plt.show()
    
    def generate_sleep_report(self, sleep_stages, metrics, patient_info=None):
        """
        Generate a comprehensive sleep analysis report.
        
        Parameters:
        -----------
        sleep_stages : dict
            Results from classify_sleep_stages()
        metrics : dict
            Results from calculate_sleep_metrics()
        patient_info : dict, optional
            Patient information for the report
        
        Returns:
        --------
        str : Formatted sleep report
        """
        if patient_info is None:
            patient_info = {
                'name': 'Anonymous',
                'age': 'N/A',
                'date': datetime.now().strftime('%Y-%m-%d')
            }
        
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    SLEEP ANALYSIS REPORT                     ║
╚═══════════════════════════════════════════════════════════════╝

Patient Information:
• Name: {patient_info['name']}
• Age: {patient_info['age']}
• Study Date: {patient_info['date']}

SLEEP SUMMARY:
═══════════════════════════════════════════════════════════════
• Total Recording Time: {metrics['total_recording_time']:.1f} minutes ({metrics['total_recording_time']/60:.1f} hours)
• Total Sleep Time: {metrics['total_sleep_time']:.1f} minutes ({metrics['total_sleep_time']/60:.1f} hours)
• Sleep Efficiency: {metrics['sleep_efficiency']:.1f}%
• Sleep Onset Latency: {metrics['sleep_onset_latency']:.1f} minutes
• REM Latency: {metrics['rem_latency']:.1f} minutes

SLEEP STAGE DISTRIBUTION:
═══════════════════════════════════════════════════════════════
"""
        
        for stage in self.sleep_stages:
            count = metrics['stage_counts'][stage]
            if stage != 'Wake':
                percentage = metrics['stage_percentages'][stage]
                report += f"• {stage}: {count} epochs ({percentage:.1f}%)\n"
            else:
                percentage = (count / sum(metrics['stage_counts'].values())) * 100
                report += f"• {stage}: {count} epochs ({percentage:.1f}%)\n"
        
        report += f"""
SLEEP ARCHITECTURE ANALYSIS:
═══════════════════════════════════════════════════════════════
• Sleep Fragmentation Index: {metrics['fragmentation_index']:.2f} transitions/hour
• Number of Stage Transitions: {metrics['transitions']}

SLEEP QUALITY ASSESSMENT:
═══════════════════════════════════════════════════════════════
• Overall Sleep Quality Score: {metrics['quality_score']:.1f}/100

Quality Interpretation:
"""
        if metrics['quality_score'] >= 80:
            report += "• Excellent sleep quality"
        elif metrics['quality_score'] >= 70:
            report += "• Good sleep quality"
        elif metrics['quality_score'] >= 60:
            report += "• Fair sleep quality - some areas for improvement"
        else:
            report += "• Poor sleep quality - significant issues detected"
        
        report += f"""

CLINICAL OBSERVATIONS:
═══════════════════════════════════════════════════════════════
• Sleep Efficiency: {"Normal" if metrics['sleep_efficiency'] >= 85 else "Below normal" if metrics['sleep_efficiency'] >= 75 else "Poor"}
• Deep Sleep (N3): {"Adequate" if metrics['stage_percentages']['N3'] >= 15 else "Insufficient"}
• REM Sleep: {"Normal" if 20 <= metrics['stage_percentages']['REM'] <= 25 else "Atypical"}
• Sleep Onset: {"Normal" if metrics['sleep_onset_latency'] <= 30 else "Delayed"}

RECOMMENDATIONS:
═══════════════════════════════════════════════════════════════
"""
        
        if metrics['sleep_efficiency'] < 85:
            report += "• Consider sleep hygiene improvements to increase sleep efficiency\n"
        if metrics['stage_percentages']['N3'] < 15:
            report += "• Insufficient deep sleep - evaluate for sleep disorders\n"
        if metrics['sleep_onset_latency'] > 30:
            report += "• Delayed sleep onset - consider relaxation techniques\n"
        if metrics['fragmentation_index'] > 15:
            report += "• High sleep fragmentation - investigate potential causes\n"
        
        report += """
Note: This analysis is based on automated EEG analysis and should be 
interpreted by qualified sleep medicine professionals.

Generated by Sleep Analysis Tool v1.0
"""
        
        return report
    
    def analyze_sleep(self, eeg_data, patient_info=None, save_plots=True):
        """
        Complete sleep analysis pipeline.
        
        Parameters:
        -----------
        eeg_data : dict
            EEG data from load_eeg_data()
        patient_info : dict, optional
            Patient information
        save_plots : bool
            Whether to save visualization plots
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print("🔬 Starting comprehensive sleep analysis...")
        
        # Step 1: Classify sleep stages
        print("\n1. Classifying sleep stages...")
        sleep_stages = self.classify_sleep_stages(eeg_data)
        
        # Step 2: Calculate metrics
        print("2. Calculating sleep metrics...")
        metrics = self.calculate_sleep_metrics(sleep_stages)
        
        # Step 3: Generate visualizations
        print("3. Creating visualizations...")
        plot_path = 'sleep_analysis_plots.png' if save_plots else None
        self.visualize_sleep_patterns(sleep_stages, metrics, plot_path)
        
        # Step 4: Generate report
        print("4. Generating report...")
        report = self.generate_sleep_report(sleep_stages, metrics, patient_info)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Sleep Quality Score: {metrics['quality_score']:.1f}/100")
        print(f"Sleep Efficiency: {metrics['sleep_efficiency']:.1f}%")
        print(f"Total Sleep Time: {metrics['total_sleep_time']/60:.1f} hours")
        print("="*60)
        
        results = {
            'sleep_stages': sleep_stages,
            'metrics': metrics,
            'report': report,
            'quality_score': metrics['quality_score']
        }
        
        return results

def demo_sleep_analysis():
    """
    Demonstrate the sleep analysis tool with synthetic data.
    """
    print("🌙 Sleep and Dream Pattern Analysis Tool Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SleepAnalyzer()
    
    # Create synthetic EEG data for demonstration
    print("Creating synthetic EEG data for demonstration...")
    duration_hours = 8
    duration_samples = int(duration_hours * 3600 * analyzer.sampling_rate)
    
    # Generate realistic sleep-like EEG signals
    t = np.linspace(0, duration_hours * 3600, duration_samples)
    
    # Create two-channel synthetic EEG with sleep-like patterns
    channel1 = np.zeros(duration_samples)
    channel2 = np.zeros(duration_samples)
    
    # Simulate different sleep stages throughout the night
    for i, time_point in enumerate(t):
        hour = time_point / 3600
        
        if hour < 0.5:  # Sleep onset - mixed wake/N1
            channel1[i] = (np.random.randn() * 0.3 + 
                          0.5 * np.sin(2 * np.pi * 10 * time_point))
            channel2[i] = (np.random.randn() * 0.3 + 
                          0.5 * np.sin(2 * np.pi * 8 * time_point))
        elif hour < 2:  # Early sleep - N2/N3
            channel1[i] = (np.random.randn() * 0.2 + 
                          2 * np.sin(2 * np.pi * 1.5 * time_point))
            channel2[i] = (np.random.randn() * 0.2 + 
                          1.5 * np.sin(2 * np.pi * 2 * time_point))
        elif hour < 4:  # Deep sleep - N3
            channel1[i] = (np.random.randn() * 0.15 + 
                          3 * np.sin(2 * np.pi * 1 * time_point))
            channel2[i] = (np.random.randn() * 0.15 + 
                          2.5 * np.sin(2 * np.pi * 0.8 * time_point))
        elif hour < 6:  # REM sleep
            channel1[i] = (np.random.randn() * 0.4 + 
                          0.8 * np.sin(2 * np.pi * 8 * time_point))
            channel2[i] = (np.random.randn() * 0.4 + 
                          0.6 * np.sin(2 * np.pi * 12 * time_point))
        else:  # Light sleep and wake
            channel1[i] = (np.random.randn() * 0.35 + 
                          np.sin(2 * np.pi * 15 * time_point))
            channel2[i] = (np.random.randn() * 0.35 + 
                          0.8 * np.sin(2 * np.pi * 20 * time_point))
    
    # Create synthetic EEG data structure
    synthetic_data = {
        'data': np.array([channel1, channel2]),
        'channel_names': ['EEG1', 'EEG2'],
        'sampling_rate': analyzer.sampling_rate,
        'duration': duration_hours * 3600
    }
    
    print(f"✓ Created {duration_hours}h synthetic EEG recording")
    
    # Patient information for demo
    patient_info = {
        'name': 'Demo Patient',
        'age': '30',
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Run complete analysis
    results = analyzer.analyze_sleep(synthetic_data, patient_info)
    
    # Save and display report
    report_path = 'sleep_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(results['report'])
    
    print(f"\n✓ Full report saved to {report_path}")
    print(f"✓ Visualization saved to sleep_analysis_plots.png")
    
    # Display abbreviated report
    print(results['report'])
    
    return results

if __name__ == "__main__":
    # Run demonstration
    demo_results = demo_sleep_analysis()
    
    print("\n🎉 Sleep analysis demonstration complete!")
    print("Check the generated files:")
    print("- sleep_analysis_report.txt")
    print("- sleep_analysis_plots.png")