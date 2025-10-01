import os
import uuid
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
from pytube import YouTube

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset'
PLOT_FOLDER = 'static/plots'
FEATURES_CSV = 'static/features.csv'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Create directories
for folder in [UPLOAD_FOLDER, PLOT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Fix matplotlib backend
plt.switch_backend('Agg')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(y, sr):
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if pitches.any() else 0

        features = np.hstack([mfcc, chroma, tempo, zcr, rms, centroid, rolloff, bandwidth, pitch])
        return features
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None

def load_dataset_features():
    if os.path.exists(FEATURES_CSV):
        try:
            df = pd.read_csv(FEATURES_CSV)
            if not df.empty:
                return df
        except:
            pass

    features = []
    for file in os.listdir(DATASET_FOLDER):
        if not allowed_file(file):
            continue
        path = os.path.join(DATASET_FOLDER, file)
        try:
            y, sr = librosa.load(path, duration=30)
            feat = extract_features(y, sr)
            if feat is not None:
                features.append([file] + feat.tolist())
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if not features:
        raise ValueError("No valid audio files found in dataset")

    columns = ['filename'] + [f'mfcc{i}' for i in range(20)] + [f'chroma{i}' for i in range(12)] + [
        'tempo', 'zcr', 'rms', 'centroid', 'rolloff', 'bandwidth', 'pitch']
    df = pd.DataFrame(features, columns=columns)

    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != 'filename']
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df.to_csv(FEATURES_CSV, index=False)
    return df

def compute_similarity(df, query_vec, weights):
    feature_cols = [col for col in df.columns if col != 'filename']
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])

    weight_array = np.array(
        [weights['w_mfcc']] * 20 +
        [weights['w_chroma']] * 12 +
        [weights['w_tempo'], weights['w_zcr'], weights['w_rms'],
         weights['w_centroid'], weights['w_rolloff'],
         weights['w_bandwidth'], weights['w_pitch']]
    )

    query_scaled = scaler.transform([query_vec]) * weight_array
    dataset_scaled = scaler.transform(df[feature_cols]) * weight_array

    sims = cosine_similarity(query_scaled, dataset_scaled)[0]
    distances = np.linalg.norm(dataset_scaled - query_scaled, axis=1)

    results = df.copy()
    results['similarity'] = sims
    results['distance'] = distances
    return results.sort_values(by='similarity', ascending=False)

def plot_features(query_vec, top_vec, plot_id):
    sns.set_theme()
    sns.set_palette("husl")
    labels = ['MFCC', 'Chroma', 'Tempo', 'ZCR', 'RMS', 'Centroid', 'Rolloff', 'Bandwidth', 'Pitch']
    q = [np.mean(query_vec[0:20]), np.mean(query_vec[20:32])] + query_vec[32:].tolist()
    t = [np.mean(top_vec[0:20]), np.mean(top_vec[20:32])] + top_vec[32:].tolist()
    x = np.arange(len(labels))

    # Bar
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, q, width=0.4, label='Query', alpha=0.8)
    plt.bar(x + 0.2, t, width=0.4, label='Top Match', alpha=0.8)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title('Feature Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/{plot_id}_bar.png', dpi=150)
    plt.close()

    # Radar
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    q += q[:1]; t += t[:1]; angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, q, 'o-', linewidth=2, label='Query')
    ax.plot(angles, t, 'o-', linewidth=2, label='Top Match')
    ax.fill(angles, q, alpha=0.2); ax.fill(angles, t, alpha=0.2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    plt.title('Feature Radar Chart')
    plt.savefig(f'{PLOT_FOLDER}/{plot_id}_radar.png', dpi=150)
    plt.close()

    # Heatmap
    plt.figure(figsize=(12, 3))
    sns.heatmap([q[:-1], t[:-1]], annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=['Query', 'Match'], cmap='YlOrRd')
    plt.title('Feature Comparison Heatmap')
    plt.tight_layout()
    plt.savefig(f'{PLOT_FOLDER}/{plot_id}_heatmap.png', dpi=150)
    plt.close()

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.vstack([query_vec, top_vec]))
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[0, 0], reduced[0, 1], s=200, label='Query', alpha=0.7)
    plt.scatter(reduced[1, 0], reduced[1, 1], s=200, label='Top Match', alpha=0.7)
    plt.legend(); plt.title('2D PCA Projection')
    plt.savefig(f'{PLOT_FOLDER}/{plot_id}_pca.png', dpi=150)
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    default_weights = {key: 1.0 for key in [
        'w_mfcc', 'w_chroma', 'w_tempo', 'w_zcr', 'w_rms',
        'w_centroid', 'w_rolloff', 'w_bandwidth', 'w_pitch'
    ]}
    if request.method == 'POST':
        try:
            weights = {key: float(request.form.get(key, 1.0)) for key in default_weights}
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(UPLOAD_FOLDER, unique_name)
                file.save(filepath)

                y, sr = librosa.load(filepath, duration=30)
                query_vec = extract_features(y, sr)
                if query_vec is None:
                    raise ValueError("Feature extraction failed")

                df = load_dataset_features()
                ranked = compute_similarity(df.copy(), query_vec, weights)
                top_result = ranked.iloc[0]
                top_vec = df[df['filename'] == top_result['filename']].drop(columns=['filename']).values[0]
                plot_id = uuid.uuid4().hex
                plot_features(query_vec, top_vec, plot_id)

                return render_template('results.html',
                                       query=unique_name,
                                       results=ranked.head(5).to_dict('records'),
                                       plot_id=plot_id,
                                       weights=weights)
        except Exception as e:
            return render_template('errors.html', error=str(e))

    return render_template('index.html', weights=default_weights)

@app.route('/youtube', methods=['POST'])
def youtube_extract():
    try:
        url = request.form['youtube_url']
        if not url:
            raise ValueError("No URL provided")
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise ValueError("No audio stream found")

        # Generate filename correctly
        filename = f"{uuid.uuid4().hex}.mp4"
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        stream.download(output_path=UPLOAD_FOLDER, filename=filename)

        audio = AudioSegment.from_file(temp_path)
        mp3_path = temp_path.replace(".mp4", ".mp3")
        audio.export(mp3_path, format="mp3")
        os.remove(temp_path)

        return redirect(url_for('index'))
    except Exception as e:
        return render_template('errors.html', error=str(e))

@app.route('/download_csv')
def download_csv():
    return send_file(FEATURES_CSV, as_attachment=True)

@app.route('/static/<path:filename>')
def static_file(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
