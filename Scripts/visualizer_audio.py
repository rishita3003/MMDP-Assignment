import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

class RadioGenreClassifier:
    """
    RadioSonicArchive Genre Classifier
    
    Use case for the RadioSonicArchive dataset: 
    Classifies radio streams into music genres based on audio features.
    """
    
    def __init__(self, dataset_dir=r".\Task1_c\radio_sonic_archive"):
        """Initialize the genre classifier"""
        self.dataset_dir = dataset_dir
        self.metadata_file = os.path.join(dataset_dir, "metadata.csv")
        self.metadata = None
        self.features = None
        self.model = None
        
        # Define genre mapping for SomaFM stations
        # This could be extended with more stations and genres
        self.genre_mapping = {
            "SomaFM_Groove": "Electronic",
            "SomaFM_Drone": "Ambient",
            "SomaFM_Jazz": "Jazz",
            "SomaFM_Indie": "Indie",
            "SomaFM_Folk": "Folk",
            "SomaFM_Ambient": "Ambient",
            "SomaFM_Lounge": "Electronic",
            "NPR_KCRW": "Eclectic",
            "BBC_Radio3": "Classical",
            "WNYC": "Talk"
        }
        
        # Load metadata if it exists
        if os.path.exists(self.metadata_file):
            self.metadata = pd.read_csv(self.metadata_file)
            print(f"Loaded metadata with {len(self.metadata)} records")
        else:
            print(f"Warning: Metadata file not found at {self.metadata_file}")
    
    def load_metadata(self, file_path=None):
        """Load metadata from CSV file"""
        if file_path is None:
            file_path = self.metadata_file
            
        self.metadata = pd.read_csv(file_path)
        print(f"Loaded metadata with {len(self.metadata)} records")
        
        # Add genre based on station name
        self.metadata['genre'] = self.metadata['station_name'].map(self.genre_mapping)
        self.metadata['genre'] = self.metadata['genre'].fillna('Unknown')
        
        return self.metadata
    
    def extract_features(self, audio_file, sr=22050, duration=None):
        """Extract audio features from a single audio file"""
        # Load audio file
        try:
            # Load only a portion if duration is specified
            if duration:
                y, sr = librosa.load(audio_file, sr=sr, duration=duration)
            else:
                y, sr = librosa.load(audio_file, sr=sr)
                
            # Feature 1: Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Feature 2: Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Feature 3: Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            # Feature 4: Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Feature 5: MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Feature 6: Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Feature 7: Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Feature 8: RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Extract global statistics from each feature
            features = {
                # Spectral features
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                
                # Time-domain features
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'zero_crossing_rate_std': np.std(zero_crossing_rate),
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                
                # Tempo
                'tempo': tempo
            }
            
            # Add MFCC statistics
            for i in range(13):
                features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
            
            # Add chroma statistics
            for i in range(12):
                features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
                features[f'chroma{i+1}_std'] = np.std(chroma[i])
                
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {str(e)}")
            return None
    
    def extract_all_features(self, max_files=None, duration=30):
        """Extract features from all audio files in the dataset"""
        if self.metadata is None:
            self.load_metadata()
            
        features_list = []
        processed_count = 0
        
        # Limit the number of files if specified
        files_to_process = self.metadata
        if max_files:
            files_to_process = self.metadata.head(max_files)
            
        total_files = len(files_to_process)
        
        for idx, row in files_to_process.iterrows():
            file_path = os.path.join(self.dataset_dir, row['filename'])
            
            if os.path.exists(file_path):
                print(f"Processing file {processed_count+1}/{total_files}: {row['filename']}")
                
                # Extract features
                audio_features = self.extract_features(file_path, duration=duration)
                
                if audio_features:
                    # Add metadata
                    audio_features['file_id'] = row['file_id']
                    audio_features['filename'] = row['filename']
                    audio_features['station_name'] = row['station_name']
                    audio_features['genre'] = row.get('genre', self.genre_mapping.get(row['station_name'], 'Unknown'))
                    
                    features_list.append(audio_features)
                    processed_count += 1
            else:
                print(f"File not found: {file_path}")
        
        # Convert to DataFrame
        self.features = pd.DataFrame(features_list)
        
        # Save features
        features_file = os.path.join(self.dataset_dir, 'audio_features.csv')
        self.features.to_csv(features_file, index=False)
        print(f"Extracted features for {len(self.features)} files")
        print(f"Features saved to {features_file}")
        
        return self.features
    
    def load_features(self, file_path=None):
        """Load pre-extracted features and fix string representation issues"""
        if file_path is None:
            file_path = os.path.join(self.dataset_dir, 'audio_features.csv')
                
        if os.path.exists(file_path):
            # First load the CSV
            self.features = pd.read_csv(file_path)
            
            # Fix string representation of lists issue
            for col in self.features.columns:
                # Skip non-numeric columns
                if col in ['file_id', 'filename', 'station_name', 'genre']:
                    continue
                    
                # Check if column contains string values
                if self.features[col].dtype == 'object':
                    # Try to clean and convert the values
                    try:
                        # Remove brackets and convert to float
                        self.features[col] = self.features[col].astype(str).str.strip('[]').astype(float)
                        print(f"Fixed column '{col}' by converting from string to float")
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' to numeric: {str(e)}")
            
            print(f"Loaded features for {len(self.features)} files")
            return self.features
        else:
            print(f"Features file not found: {file_path}")
            return None
    
    def train_model(self, test_size=0.3, random_state=42):
        """Train a genre classification model"""
        if self.features is None:
            print("No features available. Load or extract features first.")
            return None
            
        # Remove rows with unknown genre
        df = self.features[self.features['genre'] != 'Unknown'].copy()
        
        if len(df) == 0:
            print("No labeled data available for training")
            return None
            
        # Prepare feature matrix and target vector
        X = df.drop(['file_id', 'filename', 'station_name', 'genre'], axis=1, errors='ignore')
        y = df['genre']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create a pipeline with scaling and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
        
        # Train the model
        print("Training the genre classification model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        print("Evaluating the model...")
        y_pred = pipeline.predict(X_test)
        
        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save the model
        self.model = pipeline
        model_file = os.path.join(self.dataset_dir, 'genre_classifier_model.joblib')
        joblib.dump(pipeline, model_file)
        print(f"Model saved to {model_file}")
        print("Accuracy:", pipeline.score(X_test, y_test))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y.unique()),
                   yticklabels=sorted(y.unique()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Genre Classification Confusion Matrix')
        cm_file = os.path.join(self.dataset_dir, 'confusion_matrix.png')
        plt.savefig(cm_file)
        
        # Plot feature importance
        self._plot_feature_importance(pipeline, X.columns)
        
        # Return test data for further analysis
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred
        }
    
    def _plot_feature_importance(self, pipeline, feature_names):
        """Plot feature importance from the trained model"""
        if pipeline is None or not hasattr(pipeline, 'named_steps'):
            return
            
        # Get feature importance
        try:
            importances = pipeline.named_steps['classifier'].feature_importances_
            
            # Create a DataFrame for easier plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Top 20 Most Important Features for Genre Classification')
            plt.tight_layout()
            
            # Save the plot
            importance_file = os.path.join(self.dataset_dir, 'feature_importance.png')
            plt.savefig(importance_file)
            print(f"Feature importance plot saved to {importance_file}")
            
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
    
    def classify_new_audio(self, audio_file):
        """Classify a new audio file into a genre"""
        if self.model is None:
            model_file = os.path.join(self.dataset_dir, 'genre_classifier_model.joblib')
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
            else:
                print("No trained model available. Train the model first.")
                return None
        
        # Extract features from the new audio
        features = self.extract_features(audio_file, duration=30)
        
        if features is None:
            return None
            
        # Convert to DataFrame for consistency
        features_df = pd.DataFrame([features])
        
        # Drop non-feature columns if they exist
        for col in ['file_id', 'filename', 'station_name', 'genre']:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)
        
        # Make prediction
        predicted_genre = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Get class labels and their probabilities
        classes = self.model.classes_
        probs_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
        
        return {
            'file': os.path.basename(audio_file),
            'predicted_genre': predicted_genre,
            'probabilities': probs_dict
        }
    
    def visualize_audio(self, audio_file, save_dir=None):
        """Generate visualizations for an audio file"""
        if save_dir is None:
            save_dir = self.dataset_dir
            
        # Create output directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Plot waveform
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {base_name}')
            plt.tight_layout()
            waveform_file = os.path.join(save_dir, f'{base_name}_waveform.png')
            plt.savefig(waveform_file)
            plt.close()
            
            # Plot spectrogram
            plt.figure(figsize=(12, 6))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram: {base_name}')
            plt.tight_layout()
            spectrogram_file = os.path.join(save_dir, f'{base_name}_spectrogram.png')
            plt.savefig(spectrogram_file)
            plt.close()
            
            # Plot mel spectrogram
            plt.figure(figsize=(12, 6))
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram: {base_name}')
            plt.tight_layout()
            mel_file = os.path.join(save_dir, f'{base_name}_melspectrogram.png')
            plt.savefig(mel_file)
            plt.close()
            
            # Plot chromagram
            plt.figure(figsize=(12, 6))
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
            plt.colorbar()
            plt.title(f'Chromagram: {base_name}')
            plt.tight_layout()
            chroma_file = os.path.join(save_dir, f'{base_name}_chromagram.png')
            plt.savefig(chroma_file)
            plt.close()
            
            return {
                'waveform': waveform_file,
                'spectrogram': spectrogram_file,
                'mel_spectrogram': mel_file,
                'chromagram': chroma_file
            }
            
        except Exception as e:
            print(f"Error visualizing {audio_file}: {str(e)}")
            return None

def main():
    """Main function to demonstrate the genre classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RadioSonicArchive Genre Classifier")
    parser.add_argument('--dataset', type=str, required=True,
                       help="Path to the RadioSonicArchive dataset directory")
    parser.add_argument('--mode', type=str, choices=['extract', 'train', 'classify', 'visualize', 'all'],
                       default='all', help="Operation mode")
    parser.add_argument('--file', type=str, help="Audio file for classification or visualization")
    parser.add_argument('--max_files', type=int, default=None, 
                       help="Maximum number of files to process for feature extraction")
    parser.add_argument('--duration', type=int, default=30,
                       help="Duration in seconds to analyze from each audio file")
    parser.add_argument('--output', type=str, default=None,
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RadioSonicArchive Genre Classifier")
    print("=" * 50)
    
    classifier = RadioGenreClassifier(args.dataset)
    
    if args.mode == 'extract' or args.mode == 'all':
        print("\nExtracting audio features...")
        classifier.extract_all_features(max_files=args.max_files, duration=args.duration)
    
    if args.mode == 'train' or args.mode == 'all':
        print("\nTraining genre classification model...")
        if classifier.features is None:
            classifier.load_features()
        classifier.train_model()
    
    if args.mode == 'classify' and args.file:
        print(f"\nClassifying audio file: {args.file}")
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
        else:
            result = classifier.classify_new_audio(args.file)
            if result:
                print(f"\nPredicted genre: {result['predicted_genre']}")
                print("\nGenre probabilities:")
                for genre, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {genre}: {prob:.4f}")
    
    if args.mode == 'visualize' and args.file:
        print(f"\nCreating visualizations for: {args.file}")
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
        else:
            viz_results = classifier.visualize_audio(args.file, save_dir=args.output)
            if viz_results:
                print("\nVisualizations saved to:")
                for viz_type, file_path in viz_results.items():
                    print(f"  {viz_type}: {file_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()