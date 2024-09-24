from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random

app = Flask(__name__)

# Sample dataset (could be replaced with user-uploaded data)
def generate_data():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    return X

# Random Initialization (default KMeans behavior)
def kmeans_random(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='random')
    kmeans.fit(X)
    return kmeans


# Farthest First Initialization
def farthest_first_init(X, n_clusters):
    centroids = []
    centroids.append(random.choice(X))  # Select the first centroid randomly
    for _ in range(1, n_clusters):
        # Find the point that has the farthest minimum distance from the already chosen centroids
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        next_centroid = X[np.argmax(distances)]
        centroids.append(next_centroid)
    centroids = np.array(centroids)
    
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    kmeans.fit(X)
    return kmeans

# KMeans++ Initialization
def kmeans_plus_plus(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(X)
    return kmeans

# Manual Initialization (from user-selected centroids)
def kmeans_manual(X, n_clusters, initial_centroids):
    initial_centroids = np.array(initial_centroids)
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)
    kmeans.fit(X)
    return kmeans


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    data = request.json
    X = np.array(data['points'])  # Data points
    n_clusters = data['n_clusters']
    init_method = data['init_method']
    
    if init_method == 'random':
        kmeans = kmeans_random(X, n_clusters)
    elif init_method == 'farthest_first':
        kmeans = farthest_first_init(X, n_clusters)
    elif init_method == 'kmeans++':
        kmeans = kmeans_plus_plus(X, n_clusters)
    elif init_method == 'manual':
        initial_centroids = data['initial_centroids']
        kmeans = kmeans_manual(X, n_clusters, initial_centroids)
    
    # Return the resulting centroids and cluster labels
    return jsonify({
        'centroids': kmeans.cluster_centers_.tolist(),
        'labels': kmeans.labels_.tolist()
    })
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)