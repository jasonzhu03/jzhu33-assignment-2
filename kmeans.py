import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, init='random', max_iter=300, manual_centroids=None):
        """
        Initialize the KMeans clustering algorithm with parameters.
        
        Parameters:
        - n_clusters: int, number of clusters to form.
        - init: str, method for initializing centroids ('random', 'farthest', 'kmeans++', 'manual').
        - max_iter: int, maximum number of iterations for the algorithm to converge.
        - manual_centroids: array-like, optional, predefined centroids if using 'manual' initialization.
        """
        self.n_clusters = n_clusters  # Number of clusters to form
        self.init = init              # Method of initialization for centroids
        self.max_iter = max_iter      # Maximum number of iterations for convergence
        self.centroids = None         # Placeholder for the cluster centroids
        self.manual_centroids = manual_centroids  # Manual centroids, if provided
        self.labels = None            # Placeholder for labels of each point
        self.current_iter = 0         # Current iteration count

    def initialize_centroids(self, X):
        """
        Initialize the centroids based on the specified method.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - centroids: array, initialized centroids for the clusters.
        """
        if self.init == 'random':
            # Randomly select 'n_clusters' indices from the dataset
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
        elif self.init == 'farthest':
            # Use the farthest-first initialization method
            return self.farthest_first_initialization(X)
        elif self.init == 'kmeans++':
            # Use the k-means++ initialization method
            return self.kmeans_plus_plus_initialization(X)
        elif self.init == 'manual' and self.manual_centroids is not None:
            # Use manual centroids if provided
            return np.array(self.manual_centroids)
        else:
            # Raise an error if the initialization method is unsupported
            raise ValueError("Unsupported initialization method or manual centroids not provided")

    def farthest_first_initialization(self, X):
        """
        Initialize centroids using the farthest-first initialization method.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - centroids: array, initialized centroids for the clusters.
        """
        centroids = [X[np.random.choice(X.shape[0])]]  # Select the first centroid randomly
        for _ in range(1, self.n_clusters):
            # Calculate distances to the nearest centroid for each point
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            # Select the point that is farthest from any centroid
            next_centroid = X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def kmeans_plus_plus_initialization(self, X):
        """
        Initialize centroids using the k-means++ method for better initial placement.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - centroids: array, initialized centroids for the clusters.
        """
        centroids = [X[np.random.choice(X.shape[0])]]  # Select the first centroid randomly
        for _ in range(1, self.n_clusters):
            # Calculate squared distances to the nearest centroid for each point
            distances = np.min([np.linalg.norm(X - c, axis=1) ** 2 for c in centroids], axis=0)
            # Calculate probability distribution based on the distances
            prob_dist = distances / np.sum(distances)
            # Select the next centroid based on the probability distribution
            next_centroid = X[np.random.choice(X.shape[0], p=prob_dist)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - labels: array, index of the closest centroid for each data point.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # Compute distances to centroids
        return np.argmin(distances, axis=1)  # Assign labels based on the nearest centroid

    def update_centroids(self, X, labels):
        """
        Update centroids based on the current cluster assignments.
        
        Parameters:
        - X: array-like, input data to cluster.
        - labels: array, current cluster assignments for each data point.
        
        Returns:
        - new_centroids: array, updated centroids for the clusters.
        """
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def partial_fit(self, X):
        """
        Perform a single iteration of the KMeans algorithm.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - centroids: array, current centroids after fitting.
        - labels: array, current cluster assignments.
        - converged: bool, indicates if the centroids have converged.
        """
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)  # Initialize centroids if not done already

        self.labels = self.assign_clusters(X)  # Assign clusters based on current centroids
        new_centroids = self.update_centroids(X, self.labels)  # Update centroids based on the new assignments

        # Check for convergence (no change in centroids)
        if np.allclose(self.centroids, new_centroids):
            return self.centroids, self.labels, True  # Indicate convergence

        self.centroids = new_centroids  # Update centroids for the next iteration
        self.current_iter += 1  # Increment iteration count
        return self.centroids, self.labels, False  # Indicate no convergence yet

    def fit(self, X):
        """
        Fit the KMeans model to the dataset X.
        
        Parameters:
        - X: array-like, input data to cluster.
        
        Returns:
        - centroids: array, final centroids after fitting.
        - labels: array, final cluster assignments for each data point.
        """
        if self.centroids is None:
            self.centroids = self.initialize_centroids(X)  # Initialize centroids if not done already

        for _ in range(self.max_iter):
            self.labels = self.assign_clusters(X)  # Assign clusters based on current centroids
            new_centroids = self.update_centroids(X, self.labels)  # Update centroids

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                return self.centroids, self.labels  # Return final centroids and labels

            self.centroids = new_centroids  # Update centroids for the next iteration

        return self.centroids, self.labels  # Return centroids and labels after max iterations
