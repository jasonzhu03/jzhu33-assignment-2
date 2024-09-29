from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

# Variable to store the current state of the KMeans algorithm
kmeans_state = {}

@app.route('/')
def index():
    # Render the main HTML page for the KMeans application
    return render_template('index.html')

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    global kmeans_state
    data = request.json
    
    # Extract data points and parameters from the incoming JSON request
    points = np.array(data['points'])
    n_clusters = int(data['n_clusters'])
    init_method = data['init_method']
    manual_centroids = np.array(data['manual_centroids']) if 'manual_centroids' in data else None

    # Initialize KMeans with manual centroids if specified; otherwise, use default initialization
    if init_method == 'manual' and manual_centroids is not None:
        manual_centroids = np.array(data['manual_centroids'])
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, manual_centroids=manual_centroids)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init=init_method)

    # Store the KMeans state for step-wise execution
    kmeans_state = {
        'kmeans': kmeans,
        'points': points,
        'iteration': 0,
        'converged': False
    }

    # Run the first step of KMeans to initialize centroids and assign labels
    centroids, labels, converged = kmeans.partial_fit(points)
    kmeans_state['converged'] = converged

    # Return the initial centroids and labels as a JSON response
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})

@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    global kmeans_state

    # Check if the KMeans algorithm has already converged
    if kmeans_state.get('converged', False):
        return jsonify({'message': 'Already converged!', 'converged': True})

    # Retrieve the KMeans instance and data points from the state
    kmeans = kmeans_state['kmeans']
    points = kmeans_state['points']

    # Perform one step of the KMeans algorithm to update centroids and labels
    centroids, labels, converged = kmeans.partial_fit(points)

    # Update the convergence status and iteration count
    kmeans_state['converged'] = converged
    kmeans_state['iteration'] += 1

    # Return the updated centroids and labels as a JSON response
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist(), 'converged': converged})

@app.route('/run_to_convergence', methods=['POST'])
def run_to_convergence():
    global kmeans_state

    # Retrieve the KMeans instance and data points from the state
    kmeans = kmeans_state['kmeans']
    points = kmeans_state['points']

    # Run KMeans to full convergence and get the final centroids and labels
    centroids, labels = kmeans.fit(points)
    kmeans_state['converged'] = True  # Mark the algorithm as converged

    # Return the final centroids and labels as a JSON response
    return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist(), 'converged': True})

@app.route('/reset', methods=['POST'])
def reset():
    global kmeans_state
    # Clear the KMeans state to allow for a new run
    kmeans_state.clear()
    return jsonify({'message': 'KMeans state has been reset'})

if __name__ == "__main__":
    # Run the Flask app, making it accessible on all interfaces at port 3000
    app.run(host='0.0.0.0', port=3000)
