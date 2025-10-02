from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def find_similar_songs(query_features, feature_matrix, filenames, top_k=5):
    euclidean_dists = euclidean_distances([query_features], feature_matrix)[0]
    cosine_sims = cosine_similarity([query_features], feature_matrix)[0]

    euclidean_top_idx = euclidean_dists.argsort()[:top_k]
    cosine_top_idx = cosine_sims.argsort()[-top_k:][::-1]

    euclidean_results = [(filenames[i], euclidean_dists[i]) for i in euclidean_top_idx]
    cosine_results = [(filenames[i], cosine_sims[i]) for i in cosine_top_idx]

    return euclidean_results, cosine_results
