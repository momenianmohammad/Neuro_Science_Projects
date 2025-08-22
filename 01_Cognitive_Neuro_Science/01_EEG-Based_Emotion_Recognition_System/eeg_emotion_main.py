"""
EEG-Based Emotion Recognition System
===================================

A comprehensive system for detecting and classifying human emotions 
from EEG (Electroencephalogram) signals using machine learning techniques.

This system can identify four primary emotional states:
- Joy/Happiness
- Sadness  
- Stress/Anxiety
- Relaxation/Calm

Author: [Your Name]
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import mne
from scipy import signal
from scipy.stats import skew, kurtosis
import joblib
import warnings
warnings.filterwarnings('ignore')

class EEGEmotionRecognizer:
    """
    Main class for EEG-based emotion recognition system
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []
        
    def load_eeg_data(self, file_path=None, simulate=True):
        """
        Load EEG data from file or generate simulated data for demonstration
        
        Parameters:
        -----------
        file_path : str, optional
            Path to EEG data file
        simulate : bool
            Whether to generate simulated data
            
        Returns:
        --------
        data : dict
            Dictionary containing EEG data and labels
        """
        if simulate:
            print("Generating simulated EEG data...")
            return self._generate_simulated_data()
        else:
            # Code for loading real EEG data
            # This would typically use MNE-Python for .edf, .set files
            pass
    
    def _generate_simulated_data(self):
        """Generate simulated EEG data for demonstration"""
        np.random.seed(42)
        
        # EEG parameters
        n_channels = 14  # Number of EEG channels (typical for consumer devices)
        n_samples_per_trial = 1280  # 10 seconds at 128 Hz
        n_trials_per_emotion = 100
        emotions = ['joy', 'sadness', 'stress', 'relaxation']
        
        all_data = []
        all_labels = []
        
        for emotion_idx, emotion in enumerate(emotions):
            for trial in range(n_trials_per_emotion):
                # Generate frequency-specific patterns for each emotion
                eeg_trial = self._generate_emotion_specific_eeg(
                    emotion, n_channels, n_samples_per_trial
                )
                all_data.append(eeg_trial)
                all_labels.append(emotion)
        
        return {
            'data': np.array(all_data),
            'labels': np.array(all_labels),
            'channel_names': [f'EEG_{i+1}' for i in range(n_channels)],
            'sampling_rate': 128
        }
    
    def _generate_emotion_specific_eeg(self, emotion, n_channels, n_samples):
        """Generate EEG patterns specific to each emotion"""
        t = np.linspace(0, 10, n_samples)  # 10 seconds
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Base signal with noise
            base_signal = np.random.normal(0, 10, n_samples)
            
            if emotion == 'joy':
                # Higher beta (15-30 Hz) and gamma (30-100 Hz) activity
                beta = 5 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
                gamma = 3 * np.sin(2 * np.pi * 40 * t + np.random.uniform(0, 2*np.pi))
                eeg_data[ch] = base_signal + beta + gamma
                
            elif emotion == 'sadness':
                # Higher delta (0.5-4 Hz) and theta (4-8 Hz) activity
                delta = 8 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2*np.pi))
                theta = 6 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
                eeg_data[ch] = base_signal + delta + theta
                
            elif emotion == 'stress':
                # Higher beta activity with irregular patterns
                beta = 7 * np.sin(2 * np.pi * 25 * t + np.random.uniform(0, 2*np.pi))
                irregular = 4 * np.random.normal(0, 1, n_samples)
                eeg_data[ch] = base_signal + beta + irregular
                
            elif emotion == 'relaxation':
                # Higher alpha (8-13 Hz) activity
                alpha = 10 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
                eeg_data[ch] = base_signal + alpha
        
        return eeg_data
    
    def extract_features(self, eeg_data, sampling_rate=128):
        """
        Extract comprehensive features from EEG data
        
        Parameters:
        -----------
        eeg_data : array
            EEG data of shape (n_trials, n_channels, n_samples)
        sampling_rate : int
            Sampling rate of EEG data
            
        Returns:
        --------
        features : array
            Extracted features for each trial
        """
        print("Extracting features from EEG data...")
        n_trials = eeg_data.shape[0]
        all_features = []
        
        for trial_idx in range(n_trials):
            trial_data = eeg_data[trial_idx]
            trial_features = []
            
            for ch_idx in range(trial_data.shape[0]):
                channel_data = trial_data[ch_idx]
                
                # Time domain features
                trial_features.extend([
                    np.mean(channel_data),           # Mean
                    np.std(channel_data),            # Standard deviation
                    skew(channel_data),              # Skewness
                    kurtosis(channel_data),          # Kurtosis
                    np.var(channel_data),            # Variance
                    np.max(channel_data),            # Maximum
                    np.min(channel_data)             # Minimum
                ])
                
                # Frequency domain features
                freqs, psd = signal.welch(channel_data, sampling_rate, nperseg=256)
                
                # Power in different frequency bands
                delta_power = np.trapz(psd[(freqs >= 0.5) & (freqs < 4)])
                theta_power = np.trapz(psd[(freqs >= 4) & (freqs < 8)])
                alpha_power = np.trapz(psd[(freqs >= 8) & (freqs < 13)])
                beta_power = np.trapz(psd[(freqs >= 13) & (freqs < 30)])
                gamma_power = np.trapz(psd[(freqs >= 30) & (freqs < 100)])
                
                trial_features.extend([
                    delta_power, theta_power, alpha_power, beta_power, gamma_power
                ])
                
                # Relative power features
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                if total_power > 0:
                    trial_features.extend([
                        delta_power / total_power,
                        theta_power / total_power,
                        alpha_power / total_power,
                        beta_power / total_power,
                        gamma_power / total_power
                    ])
                else:
                    trial_features.extend([0, 0, 0, 0, 0])
            
            all_features.append(trial_features)
        
        # Generate feature names
        if not self.feature_names:
            self.feature_names = []
            for ch in range(trial_data.shape[0]):
                ch_name = f'CH{ch+1}'
                self.feature_names.extend([
                    f'{ch_name}_mean', f'{ch_name}_std', f'{ch_name}_skew',
                    f'{ch_name}_kurt', f'{ch_name}_var', f'{ch_name}_max', f'{ch_name}_min',
                    f'{ch_name}_delta', f'{ch_name}_theta', f'{ch_name}_alpha',
                    f'{ch_name}_beta', f'{ch_name}_gamma',
                    f'{ch_name}_rel_delta', f'{ch_name}_rel_theta', f'{ch_name}_rel_alpha',
                    f'{ch_name}_rel_beta', f'{ch_name}_rel_gamma'
                ])
        
        return np.array(all_features)
    
    def train_model(self, features, labels, model_type='rf'):
        """
        Train emotion classification model
        
        Parameters:
        -----------
        features : array
            Feature matrix
        labels : array
            Emotion labels
        model_type : str
            Type of model ('rf' for Random Forest, 'svm' for SVM)
        """
        print(f"Training {model_type.upper()} model...")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        # Train model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_
        ))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, features_scaled, labels_encoded, cv=5)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return X_test, y_test, y_pred
    
    def predict_emotion(self, eeg_data, sampling_rate=128):
        """
        Predict emotion from new EEG data
        
        Parameters:
        -----------
        eeg_data : array
            EEG data of shape (n_channels, n_samples)
        sampling_rate : int
            Sampling rate
            
        Returns:
        --------
        prediction : str
            Predicted emotion
        confidence : float
            Prediction confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Add batch dimension if single trial
        if len(eeg_data.shape) == 2:
            eeg_data = eeg_data[np.newaxis, ...]
        
        # Extract features
        features = self.extract_features(eeg_data, sampling_rate)
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence (for models that support predict_proba)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0  # SVM doesn't have predict_proba by default
        
        return prediction, confidence
    
    def visualize_results(self, X_test, y_test, y_pred):
        """Create visualizations for model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_, ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # Feature Importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            axes[0,1].bar(range(20), importances[indices])
            axes[0,1].set_title('Top 20 Feature Importances')
            axes[0,1].set_xlabel('Features')
            axes[0,1].set_ylabel('Importance')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        
        scatter = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis')
        axes[1,0].set_title('PCA Visualization of Test Data')
        axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Class Distribution
        unique, counts = np.unique(y_test, return_counts=True)
        class_names = self.label_encoder.inverse_transform(unique)
        axes[1,1].bar(class_names, counts)
        axes[1,1].set_title('Test Set Class Distribution')
        axes[1,1].set_xlabel('Emotion')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('eeg_emotion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='eeg_emotion_model.joblib'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='eeg_emotion_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")

def main():
    """Main execution function"""
    print("EEG-Based Emotion Recognition System")
    print("=" * 50)
    
    # Initialize the system
    recognizer = EEGEmotionRecognizer()
    
    # Load or generate data
    data = recognizer.load_eeg_data(simulate=True)
    print(f"Loaded data: {data['data'].shape[0]} trials, {data['data'].shape[1]} channels")
    
    # Extract features
    features = recognizer.extract_features(data['data'], data['sampling_rate'])
    print(f"Extracted features: {features.shape}")
    
    # Train model
    X_test, y_test, y_pred = recognizer.train_model(features, data['labels'], model_type='rf')
    
    # Create visualizations
    recognizer.visualize_results(X_test, y_test, y_pred)
    
    # Save model
    recognizer.save_model()
    
    # Example prediction on new data
    print("\nTesting prediction on new data...")
    test_trial = data['data'][0]  # Use first trial as example
    predicted_emotion, confidence = recognizer.predict_emotion(test_trial)
    actual_emotion = data['labels'][0]
    
    print(f"Actual emotion: {actual_emotion}")
    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Confidence: {confidence:.3f}")
    
    print("\nSystem ready for emotion recognition from EEG signals!")

if __name__ == "__main__":
    main()
