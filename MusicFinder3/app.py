import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from features.extractor import extract_features, build_feature_dataset, get_feature_names, apply_weights
from utils.similarity import find_similar_songs
from utils.visualization import (
    visualize_feature_difference,
    bar_feature_difference,
    heatmap_similarity_matrix,
    radar_plot
)

st.set_page_config(page_title="🎵 Song Similarity Finder", layout="centered")

# ------------------ Caching ------------------
@st.cache_data(show_spinner="🔍 Extracting query features...")
def cached_extract_query_features(path):
    return extract_features(path)

@st.cache_data(show_spinner="📂 Extracting dataset features...")
def cached_build_feature_dataset(folder_path):
    return build_feature_dataset(folder_path)

@st.cache_resource
def cached_standard_scaler_fit(X):
    scaler = StandardScaler()
    return scaler.fit(X)

# ------------------ Sidebar ------------------
st.sidebar.markdown("""
### 🎵 Audio Similarity Finder

Upload a song and find similar ones using:
- Time & frequency domain
- MFCC, Chroma, Pitch, Tempo

Choose:
- Euclidean or Cosine similarity
- Feature normalization
- Feature weighting
""")

st.sidebar.subheader("🎛 Feature Weight Tuning")
use_weights = st.sidebar.checkbox("Enable Weighted Features", value=False)
w_time = st.sidebar.slider("Time-domain", 0.0, 2.0, 1.0)
w_freq = st.sidebar.slider("Frequency-domain", 0.0, 2.0, 1.0)
w_mfcc = st.sidebar.slider("MFCC", 0.0, 2.0, 1.0)
w_chroma = st.sidebar.slider("Chroma", 0.0, 2.0, 1.0)
w_other = st.sidebar.slider("Other", 0.0, 2.0, 1.0)
weights = [w_time, w_freq, w_mfcc, w_chroma, w_other]

# ------------------ Main UI ------------------
uploaded_file = st.file_uploader("Upload a query song (.wav or .mp3)", type=["wav", "mp3"])
dataset_folder = st.text_input("Enter path to your audio dataset folder:")

if uploaded_file and dataset_folder:
    if st.button("🔍 Find Similar Songs"):
        st.session_state['start_search'] = True

if 'start_search' in st.session_state and st.session_state['start_search']:
    with open("temp_query.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    query_features_raw = cached_extract_query_features("temp_query.wav")
    feature_matrix_raw, filenames = cached_build_feature_dataset(dataset_folder)
    feature_names = get_feature_names()

    st.subheader("Dataset Feature Preview")
    df_raw = pd.DataFrame(feature_matrix_raw, columns=feature_names)
    st.dataframe(df_raw.head(10).style.format(precision=4))

    csv_dataset = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("📂 Download Dataset Features as CSV", data=csv_dataset, file_name="dataset_features.csv")

    normalize = st.checkbox("Apply Feature Normalization")
    if normalize:
        scaler = cached_standard_scaler_fit(feature_matrix_raw)
        feature_matrix_norm = scaler.transform(feature_matrix_raw)
        query_features_norm = scaler.transform([query_features_raw])[0]

        st.subheader("Normalized Feature Preview")
        df_norm = pd.DataFrame(feature_matrix_norm, columns=feature_names)
        st.dataframe(df_norm.head(10).style.format(precision=4))

        st.subheader("Query Feature (Normalized)")
        st.dataframe(pd.DataFrame([query_features_norm], columns=feature_names).style.format(precision=4))
    else:
        feature_matrix_norm = feature_matrix_raw
        query_features_norm = query_features_raw

    if use_weights:
        matrix_used = np.array([apply_weights(f, weights) for f in feature_matrix_norm])
        query_used = apply_weights(query_features_norm, weights)
    else:
        matrix_used = feature_matrix_norm
        query_used = query_features_norm

    euclidean_results, cosine_results = find_similar_songs(query_used, matrix_used, filenames)

    euclidean_df = pd.DataFrame(euclidean_results, columns=["Filename", "Euclidean Distance"])
    cosine_df = pd.DataFrame(cosine_results, columns=["Filename", "Cosine Similarity"])

    metric_choice = st.radio("Choose similarity metric to display:", ("Euclidean Distance", "Cosine Similarity"), horizontal=True)

    if metric_choice == "Euclidean Distance":
        st.subheader("Top Matches (Euclidean Distance)")
        display_df = euclidean_df.copy().rename(columns={"Euclidean Distance": "Distance"})
    else:
        st.subheader("Top Matches (Cosine Similarity)")
        display_df = cosine_df.copy().rename(columns={"Cosine Similarity": "Similarity"})

    st.dataframe(display_df, use_container_width=True)

    st.subheader("Feature Comparison with Top Match")
    top_match_name = display_df.iloc[0]["Filename"]
    top_match_index = filenames.index(top_match_name)
    match_features = matrix_used[top_match_index]

    tabs = st.tabs(["📈 Line Plot", "📊 Bar Plot", "🌡 Heatmap", "🕸 Radar"])
    with tabs[0]:
        st.pyplot(visualize_feature_difference(query_used, match_features, feature_names))
    with tabs[1]:
        st.pyplot(bar_feature_difference(query_used, match_features, feature_names))
    with tabs[2]:
        st.pyplot(heatmap_similarity_matrix(query_used, match_features, feature_names))
    with tabs[3]:
        radar_fig = radar_plot(query_used, match_features, feature_names)
        if radar_fig:
            st.pyplot(radar_fig)
        else:
            st.warning("Too many features for radar plot. Reduce dimensionality first.")

# ------------------ Reset Button ------------------
if st.button("🔄 Clear Cache / Reset App"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    for temp_file in ["temp_query.wav"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    st.rerun()
