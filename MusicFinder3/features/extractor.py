import numpy as np
import librosa
import os

def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)

    # Time-domain
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    energy = np.sum(np.square(y)) / len(y)

    # Frequency-domain
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Time-frequency
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Pitch & Intensity
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    intensity = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 0

    return np.hstack([
        zcr, rms, energy,
        spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness,
        mfccs, chroma,
        tempo, pitch_mean, intensity
    ])

def build_feature_dataset(folder_path):
    data = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".wav", ".mp3")):
            try:
                path = os.path.join(folder_path, file)
                features = extract_features(path)
                data.append(features)
                filenames.append(file)
            except Exception as e:
                print(f"[Error] {file}: {e}")
    return np.array(data), filenames

def get_feature_names():
    return (
        ['ZCR', 'RMS', 'Energy', 'Spectral Centroid', 'Spectral Bandwidth',
         'Spectral Rolloff', 'Spectral Flatness'] +
        [f'MFCC_{i}' for i in range(13)] +
        [f'Chroma_{i}' for i in range(12)] +
        ['Tempo', 'Pitch', 'Intensity']
    )

def apply_weights(features, weights):
    """
    Apply weights to feature groups:
    weights = [time, frequency, mfcc, chroma, other]
    
    Feature index mapping (based on get_feature_names()):
    - Time-domain: indices 0–2
    - Frequency-domain: indices 3–6
    - MFCCs: indices 7–19
    - Chroma: indices 20–31
    - Other (Tempo, Pitch, Intensity): indices 32–34
    """
    weighted = np.copy(features)

    # Define slices based on feature structure
    time_slice = slice(0, 3)         # ZCR, RMS, Energy
    freq_slice = slice(3, 7)         # Spectral features
    mfcc_slice = slice(7, 20)        # MFCC_0 to MFCC_12
    chroma_slice = slice(20, 32)     # Chroma_0 to Chroma_11
    other_slice = slice(32, 35)      # Tempo, Pitch, Intensity

    # Apply group-specific weights
    weighted[time_slice] *= weights[0]
    weighted[freq_slice] *= weights[1]
    weighted[mfcc_slice] *= weights[2]
    weighted[chroma_slice] *= weights[3]
    weighted[other_slice] *= weights[4]

    return weighted

