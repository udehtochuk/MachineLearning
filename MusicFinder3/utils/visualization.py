import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_feature_difference(query, match, feature_names=None, title="Feature Comparison"):
    plt.figure(figsize=(14, 5))
    plt.plot(query, label="Query", marker='o')
    plt.plot(match, label="Match", marker='x')
    
    if feature_names:
        plt.xticks(ticks=np.arange(len(query)), labels=feature_names, rotation=90)
    
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    return plt


def bar_feature_difference(query, match, feature_names):
    diff = np.abs(query - match)
    plt.figure(figsize=(14, 5))
    sns.barplot(x=feature_names, y=diff)
    plt.xticks(rotation=90)
    plt.title("Absolute Feature Difference (Query vs Match)")
    plt.ylabel("Absolute Difference")
    plt.tight_layout()
    return plt


def heatmap_similarity_matrix(query, match, feature_names):
    sim_matrix = np.vstack([query, match])
    plt.figure(figsize=(10, 2))
    sns.heatmap(sim_matrix, annot=False, cmap='coolwarm', cbar=True,
                xticklabels=feature_names, yticklabels=["Query", "Match"])
    plt.title("Feature Heatmap (Query vs Match)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt


def radar_plot(query, match, feature_names):
    if len(query) > 30:  # Too many features = clutter
        return None

    labels = np.array(feature_names)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    query = np.append(query, query[0])
    match = np.append(match, match[0])
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, query, label='Query', marker='o')
    ax.plot(angles, match, label='Match', marker='x')
    ax.fill(angles, query, alpha=0.2)
    ax.fill(angles, match, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Radar Plot of Features", y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    return plt
