from features.extractor import extract_features, build_feature_dataset
from utils.similarity import find_similar_songs

query_file = "temp_query.wav"
dataset_folder = "data"

query_features = extract_features(query_file)
feature_matrix, filenames = build_feature_dataset(dataset_folder)

euclidean_results, cosine_results = find_similar_songs(query_features, feature_matrix, filenames)

print("Euclidean Top Matches:")
for name, score in euclidean_results:
    print(f"{name} - Distance: {score:.4f}")

print("\nCosine Top Matches:")
for name, score in cosine_results:
    print(f"{name} - Similarity: {score:.4f}")
