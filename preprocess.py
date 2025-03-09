import librosa
import numpy as np

def extract_features(file_path):
    """Faster feature extraction with MFCC only."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, mono=True)  # Use lower sample rate
    except Exception as e:
        print(f" Error loading {file_path}: {e}")
        return None

    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)  # Reduce MFCC size

    return mfccs  # Return only MFCC features (faster training)
