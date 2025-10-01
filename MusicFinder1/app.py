import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Listbox
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import threading
import pygame
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- Feature Extraction ----------
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)

    # Time domain features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Frequency domain features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Time-frequency domain features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

    # Rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Combine all features
    feature_vector = np.hstack([
        zcr,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_flatness,
        mfcc,
        chroma,
        tonnetz,
        tempo
    ])

    return feature_vector

def load_dataset(folder_path):
    features_list = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".wav", ".mp3")):
            path = os.path.join(folder_path, filename)
            try:
                features = extract_features(path)
                features_list.append(features)
                filenames.append(path)
            except Exception as e:
                print(f"Failed on {filename}: {e}")
    return np.array(features_list), filenames

# ---------- Similarity with feature weights ----------
def weighted_cosine_similarity(query_feat, dataset_feats, weights):
    sims = []
    for features in dataset_feats:
        diff = (query_feat - features) * weights
        dist = np.linalg.norm(diff)  # weighted Euclidean distance
        sims.append(dist)
    return sims

def create_feature_weights(time_w, freq_w, tf_w, rhythm_w):
    weights = np.zeros(38)
    weights[0:2] = time_w
    weights[2:6] = freq_w
    weights[6:31] = tf_w
    weights[37] = rhythm_w
    return weights

def find_similar_weighted(query_feat, dataset_feats, filenames, weights, top_n=5):
    distances = weighted_cosine_similarity(query_feat, dataset_feats, weights)
    similarities = list(zip(filenames, distances))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

# ---------- GUI ----------
class MusicSimilarityApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Music Similarity Search")

        self.dataset_path = ""
        self.dataset_feats = []
        self.filenames = []
        self.scaler = StandardScaler()

        self.query_file = ""
        self.query_feat = None
        self.top_match_feat = None
        self.top_match_file = None

        self.setup_gui()

        pygame.init()

    def setup_gui(self):
        tk.Button(self.master, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        tk.Button(self.master, text="Choose Query Audio", command=self.choose_query).pack(pady=5)
        tk.Button(self.master, text="Find Similar Songs", command=self.search_similar).pack(pady=5)

        # Feature weight sliders
        self.time_weight = tk.DoubleVar(value=1.0)
        self.freq_weight = tk.DoubleVar(value=1.0)
        self.tf_weight = tk.DoubleVar(value=1.0)
        self.rhythm_weight = tk.DoubleVar(value=1.0)

        slider_frame = tk.Frame(self.master)
        slider_frame.pack(pady=5)

        tk.Label(slider_frame, text="Time Domain Weight").grid(row=0, column=0, sticky='w')
        tk.Scale(slider_frame, variable=self.time_weight, from_=0, to=5, resolution=0.1, orient='horizontal').grid(row=0, column=1)

        tk.Label(slider_frame, text="Frequency Domain Weight").grid(row=1, column=0, sticky='w')
        tk.Scale(slider_frame, variable=self.freq_weight, from_=0, to=5, resolution=0.1, orient='horizontal').grid(row=1, column=1)

        tk.Label(slider_frame, text="Time-Frequency Domain Weight").grid(row=2, column=0, sticky='w')
        tk.Scale(slider_frame, variable=self.tf_weight, from_=0, to=5, resolution=0.1, orient='horizontal').grid(row=2, column=1)

        tk.Label(slider_frame, text="Rhythm Weight").grid(row=3, column=0, sticky='w')
        tk.Scale(slider_frame, variable=self.rhythm_weight, from_=0, to=5, resolution=0.1, orient='horizontal').grid(row=3, column=1)

        self.result_list = Listbox(self.master, width=60)
        self.result_list.pack(pady=10)
        self.result_list.bind('<Double-1>', self.play_selected_song)

        # Export CSV button
        tk.Button(self.master, text="Export Features to CSV", command=self.export_features_csv).pack(pady=5)

        # Visualization buttons
        viz_frame = tk.Frame(self.master)
        viz_frame.pack(pady=5)

        tk.Button(viz_frame, text="Show Radar Chart", command=self.show_radar_chart).grid(row=0, column=0, padx=5)
        tk.Button(viz_frame, text="Show Heatmap", command=self.show_heatmap).grid(row=0, column=1, padx=5)
        tk.Button(viz_frame, text="Show PCA Plot", command=self.show_pca_plot).grid(row=0, column=2, padx=5)

        # NEW FEATURE: Feature Differences Bar Plot button
        tk.Button(viz_frame, text="Show Feature Differences Bar Plot", command=self.show_feature_diff_bar).grid(row=0, column=3, padx=5)

    def load_dataset(self):
        folder = filedialog.askdirectory(title="Select Folder of Songs")
        if folder:
            self.dataset_feats, self.filenames = load_dataset(folder)
            self.dataset_feats = self.scaler.fit_transform(self.dataset_feats)
            self.dataset_path = folder
            messagebox.showinfo("Dataset Loaded", f"Loaded {len(self.filenames)} audio files.")

    def choose_query(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file:
            self.query_file = file
            try:
                features = extract_features(file)
                self.query_feat = self.scaler.transform([features])[0]
                messagebox.showinfo("Query Loaded", f"Selected: {os.path.basename(file)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process query: {e}")

    def search_similar(self):
        if self.query_feat is None or len(self.dataset_feats) == 0:
            messagebox.showerror("Missing Data", "Load dataset and query audio first.")
            return

        weights = create_feature_weights(
            self.time_weight.get(),
            self.freq_weight.get(),
            self.tf_weight.get(),
            self.rhythm_weight.get()
        )
        matches = find_similar_weighted(self.query_feat, self.dataset_feats, self.filenames, weights)
        self.result_list.delete(0, tk.END)
        for match in matches:
            name = os.path.basename(match[0])
            self.result_list.insert(tk.END, f"{name}  (Distance: {match[1]:.4f})")

        if matches:
            top_match_idx = self.filenames.index(matches[0][0])
            self.top_match_feat = self.dataset_feats[top_match_idx]
            self.top_match_file = matches[0][0]
        else:
            self.top_match_feat = None
            self.top_match_file = None

    def play_selected_song(self, event):
        index = self.result_list.curselection()
        if index:
            filepath = self.filenames[index[0]]
            threading.Thread(target=self.play_audio, args=(filepath,), daemon=True).start()

            if hasattr(self, 'top_match_feat') and filepath == self.top_match_file:
                self.plot_radar_chart(self.query_feat, self.top_match_feat,
                                      "Query Audio", os.path.basename(self.top_match_file))

    def play_audio(self, filepath):
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    # --- CSV Export ---
    def export_features_csv(self):
        if len(self.dataset_feats) == 0:
            messagebox.showerror("No Data", "Load dataset first.")
            return

        feature_names = self.get_feature_names()
        df = pd.DataFrame(self.dataset_feats, columns=feature_names)
        df['filename'] = self.filenames
        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")],
                                                 title="Save features CSV")
        if save_path:
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Features saved to {save_path}")

    # --- Visualization ---

    def get_feature_names(self):
        return [
            "ZCR", "RMS", "Centroid", "Bandwidth", "Rolloff", "Flatness"
        ] + [f"MFCC{i+1}" for i in range(13)] + \
            [f"Chroma{i+1}" for i in range(12)] + \
            [f"Tonnetz{i+1}" for i in range(6)] + \
            ["Tempo"]

    def show_radar_chart(self):
        index = self.result_list.curselection()
        if index and self.query_feat is not None:
            selected_file = self.filenames[index[0]]
            selected_feat = self.dataset_feats[index[0]]
            self.plot_radar_chart(self.query_feat, selected_feat, "Query Audio", os.path.basename(selected_file))
        else:
            messagebox.showerror("Selection Error", "Select a matched song and ensure query is loaded.")

    def plot_radar_chart(self, feat1, feat2, label1, label2):
        features = self.get_feature_names()
        N = len(features)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        feat1 = np.append(feat1, feat1[0])  # close the loop
        feat2 = np.append(feat2, feat2[0])
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, feat1, label=label1)
        ax.fill(angles, feat1, alpha=0.25)
        ax.plot(angles, feat2, label=label2)
        ax.fill(angles, feat2, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), features + [features[0]])
        ax.set_title("Feature Comparison (Radar Chart)")
        ax.legend(loc='upper right')
        plt.show()

    def show_heatmap(self):
        if len(self.dataset_feats) == 0:
            messagebox.showerror("No Data", "Load dataset first.")
            return
        df = pd.DataFrame(self.dataset_feats, columns=self.get_feature_names())
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def show_pca_plot(self):
        if len(self.dataset_feats) == 0:
            messagebox.showerror("No Data", "Load dataset first.")
            return
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.dataset_feats)
        plt.figure(figsize=(10, 7))
        plt.scatter(components[:, 0], components[:, 1])
        plt.title("PCA Projection of Dataset Features")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    # --- NEW FEATURE: Feature Differences Bar Plot ---
    def show_feature_diff_bar(self):
        index = self.result_list.curselection()
        if index is None or len(index) == 0:
            messagebox.showerror("Selection Error", "Select a matched song to compare.")
            return
        if self.query_feat is None:
            messagebox.showerror("Missing Query", "Load and select a query audio first.")
            return

        selected_idx = index[0]
        selected_feat = self.dataset_feats[selected_idx]
        diff = np.abs(self.query_feat - selected_feat)

        feature_names = self.get_feature_names()

        plt.figure(figsize=(14, 6))
        plt.bar(range(len(diff)), diff, color='skyblue')
        plt.xticks(range(len(diff)), feature_names, rotation=90)
        plt.ylabel("Absolute Feature Difference")
        plt.title(f"Feature Differences between Query and {os.path.basename(self.filenames[selected_idx])}")
        plt.tight_layout()
        plt.show()

# ---------- Run App ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicSimilarityApp(root)
    root.mainloop()
