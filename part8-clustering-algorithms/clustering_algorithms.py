import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import random
import math


class KMeans:
    """
    K-Means clustering algorithm implementation.
    Partitions data into k clusters by minimizing within-cluster sum of squares.
    """
    
    def __init__(self, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4, 
                 init_method: str = 'random'):
        """
        Initialize K-Means parameters.
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            init_method: Initialization method ('random', 'kmeans++')
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.init_method = init_method
        self.centroids = None
        self.labels = None
        self.history = []
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit K-Means clustering to data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids
        if self.init_method == 'kmeans++':
            self.centroids = self._init_kmeans_plus_plus(X)
        else:
            self.centroids = self._init_random(X)
        
        # Main clustering loop
        for iteration in range(self.max_iterations):
            # Store previous centroids for convergence check
            prev_centroids = self.centroids.copy()
            
            # Assign points to nearest centroids
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[i] = self.centroids[i]
            
            self.centroids = new_centroids
            
            # Store history for analysis
            inertia = self._compute_inertia(X)
            self.history.append({
                'iteration': iteration,
                'centroids': self.centroids.copy(),
                'inertia': inertia
            })
            
            # Check for convergence
            centroid_shift = np.mean(np.sqrt(np.sum((self.centroids - prev_centroids) ** 2, axis=1)))
            if centroid_shift < self.tolerance:
                print(f"K-Means converged after {iteration + 1} iterations")
                break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _init_random(self, X: np.ndarray) -> np.ndarray:
        """Random centroid initialization."""
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        
        for i in range(n_features):
            min_val, max_val = X[:, i].min(), X[:, i].max()
            centroids[:, i] = np.random.uniform(min_val, max_val, self.k)
        
        return centroids
    
    def _init_kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization for better centroid selection."""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        # Choose remaining centroids
        for i in range(1, self.k):
            # Compute distances to nearest centroid
            distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]]) for x in X])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
        
        return centroids
    
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between points and centroids."""
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def _compute_inertia(self, X: np.ndarray) -> float:
        """Compute within-cluster sum of squares (inertia)."""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.
    Groups together points that are closely packed while marking outliers.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN parameters.
        
        Args:
            eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples: Number of samples in a neighborhood for a point to be considered as a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_samples = None
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit DBSCAN clustering to data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)  # Initialize all as noise (-1)
        self.core_samples = set()
        
        # Find neighbors for each point
        neighbors = [self._region_query(X, i) for i in range(n_samples)]
        
        cluster_id = 0
        
        for i in range(n_samples):
            # Skip if already processed
            if self.labels[i] != -1:
                continue
            
            # Check if point has enough neighbors to be a core point
            if len(neighbors[i]) < self.min_samples:
                continue
            
            # Start new cluster
            self.core_samples.add(i)
            self.labels[i] = cluster_id
            
            # Expand cluster
            seed_set = list(neighbors[i])
            j = 0
            
            while j < len(seed_set):
                point = seed_set[j]
                
                # Change noise to border point
                if self.labels[point] == -1:
                    self.labels[point] = cluster_id
                
                # If not yet processed
                if self.labels[point] == -1:
                    self.labels[point] = cluster_id
                    
                    # If point is core point, add its neighbors to seed set
                    if len(neighbors[point]) >= self.min_samples:
                        self.core_samples.add(point)
                        seed_set.extend(neighbors[point])
                
                j += 1
            
            cluster_id += 1
    
    def _region_query(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Find all points within eps distance of given point."""
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[point_idx] - X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Note: DBSCAN doesn't have a traditional predict method.
        This assigns new points to nearest cluster or marks as noise.
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # For simplicity, assign to nearest core point's cluster
        predictions = np.full(X.shape[0], -1)
        
        for i, point in enumerate(X):
            min_distance = float('inf')
            closest_cluster = -1
            
            for core_idx in self.core_samples:
                distance = np.linalg.norm(point - self.training_data[core_idx])
                if distance <= self.eps and distance < min_distance:
                    min_distance = distance
                    closest_cluster = self.labels[core_idx]
            
            predictions[i] = closest_cluster
        
        return predictions


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering implementation.
    Builds tree of clusters by iteratively merging closest clusters.
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'single'):
        """
        Initialize Hierarchical Clustering parameters.
        
        Args:
            n_clusters: Number of clusters to form
            linkage: Linkage criterion ('single', 'complete', 'average')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.dendrogram = []
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit hierarchical clustering to data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        
        # Initialize: each point is its own cluster
        clusters = [[i] for i in range(n_samples)]
        cluster_centers = X.copy()
        
        # Merge clusters until we have desired number
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(
                        X, clusters[i], clusters[j]
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Merge closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
            
            # Store merge information for dendrogram
            self.dendrogram.append({
                'clusters': len(clusters) + 1,
                'distance': min_distance,
                'merged': (merge_i, merge_j)
            })
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_points in enumerate(clusters):
            for point in cluster_points:
                self.labels[point] = cluster_id
    
    def _cluster_distance(self, X: np.ndarray, cluster1: List[int], cluster2: List[int]) -> float:
        """Compute distance between two clusters based on linkage criterion."""
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        if self.linkage == 'single':
            # Minimum distance between any two points
            min_dist = float('inf')
            for p1 in points1:
                for p2 in points2:
                    dist = np.linalg.norm(p1 - p2)
                    min_dist = min(min_dist, dist)
            return min_dist
        
        elif self.linkage == 'complete':
            # Maximum distance between any two points
            max_dist = 0
            for p1 in points1:
                for p2 in points2:
                    dist = np.linalg.norm(p1 - p2)
                    max_dist = max(max_dist, dist)
            return max_dist
        
        elif self.linkage == 'average':
            # Average distance between all pairs of points
            total_dist = 0
            count = 0
            for p1 in points1:
                for p2 in points2:
                    total_dist += np.linalg.norm(p1 - p2)
                    count += 1
            return total_dist / count
        
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")


class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) using Expectation-Maximization algorithm.
    Models data as a mixture of Gaussian distributions.
    """
    
    def __init__(self, n_components: int = 2, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Initialize GMM parameters.
        
        Args:
            n_components: Number of Gaussian components
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = None  # Mixture weights
        self.means = None    # Component means
        self.covariances = None  # Component covariances
        self.log_likelihood_history = []
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit GMM to data using EM algorithm.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step: Compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: Update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check for convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"GMM converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster labels (most likely component)
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster probabilities for new data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            Cluster probabilities
        """
        return self._e_step(X)
    
    def _initialize_parameters(self, X: np.ndarray) -> None:
        """Initialize GMM parameters using K-means."""
        n_samples, n_features = X.shape
        
        # Use K-means for initialization
        kmeans = KMeans(k=self.n_components, init_method='kmeans++')
        kmeans.fit(X)
        
        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = kmeans.centroids
        
        # Initialize covariances
        self.covariances = []
        for i in range(self.n_components):
            cluster_points = X[kmeans.labels == i]
            if len(cluster_points) > 1:
                cov = np.cov(cluster_points.T)
            else:
                cov = np.eye(n_features)
            self.covariances.append(cov)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: Compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = (self.weights[k] * 
                                    self._multivariate_gaussian(X, self.means[k], self.covariances[k]))
        
        # Normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """M-step: Update parameters."""
        n_samples, n_features = X.shape
        
        # Update weights
        N_k = np.sum(responsibilities, axis=0)
        self.weights = N_k / n_samples
        
        # Update means
        self.means = np.dot(responsibilities.T, X) / N_k.reshape(-1, 1)
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k])
            
            # Add small regularization to prevent singular matrices
            self.covariances[k] += 1e-6 * np.eye(n_features)
    
    def _multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute multivariate Gaussian probability density."""
        n_features = X.shape[1]
        diff = X - mean
        
        # Add regularization to prevent numerical issues
        cov_reg = cov + 1e-6 * np.eye(n_features)
        
        try:
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to identity if matrix is singular
            inv_cov = np.eye(n_features)
            det_cov = 1.0
        
        exponent = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * abs(det_cov))
        
        return normalization * np.exp(exponent)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data."""
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += (self.weights[k] * 
                                    self._multivariate_gaussian(X[i:i+1], self.means[k], self.covariances[k]))
            log_likelihood += np.log(sample_likelihood + 1e-10)  # Add small epsilon
        
        return log_likelihood


def generate_cluster_data(n_samples: int = 300, n_centers: int = 3, 
                         cluster_std: float = 1.0, center_box: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustering data.
    
    Args:
        n_samples: Total number of samples
        n_centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        center_box: Bounding box for cluster centers
        
    Returns:
        X: Data points, y: True cluster labels
    """
    # Generate cluster centers
    centers = np.random.uniform(center_box[0], center_box[1], (n_centers, 2))
    
    # Generate points around each center
    samples_per_cluster = n_samples // n_centers
    X = []
    y = []
    
    for i, center in enumerate(centers):
        # Generate samples for this cluster
        cluster_samples = np.random.normal(center, cluster_std, (samples_per_cluster, 2))
        X.append(cluster_samples)
        y.extend([i] * samples_per_cluster)
    
    # Add remaining samples to last cluster if needed
    remaining = n_samples - len(y)
    if remaining > 0:
        last_center = centers[-1]
        additional_samples = np.random.normal(last_center, cluster_std, (remaining, 2))
        X.append(additional_samples)
        y.extend([n_centers - 1] * remaining)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate clustering performance using various metrics.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Simple accuracy (assuming cluster labels can be matched)
    # This is a simplified version - in practice, you'd use more sophisticated metrics
    
    # Count matches for each possible label mapping
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    best_accuracy = 0
    
    # Try different label mappings (brute force for small number of clusters)
    from itertools import permutations
    
    for perm in permutations(unique_pred):
        if len(perm) != len(unique_true):
            continue
        
        # Create mapping
        mapping = dict(zip(perm, unique_true))
        mapped_pred = np.array([mapping.get(label, -1) for label in y_pred])
        
        accuracy = np.mean(mapped_pred == y_true)
        best_accuracy = max(best_accuracy, accuracy)
    
    return {'accuracy': best_accuracy}


if __name__ == "__main__":
    # Example usage and comparison of clustering algorithms
    print("Clustering Algorithms Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic clustering data...")
    X, y_true = generate_cluster_data(n_samples=300, n_centers=3, cluster_std=1.5)
    
    print(f"Generated {len(X)} samples with {len(np.unique(y_true))} true clusters")
    
    # Test different clustering algorithms
    algorithms = {
        'K-Means': KMeans(k=3, init_method='kmeans++'),
        'K-Means (Random Init)': KMeans(k=3, init_method='random'),
        'DBSCAN': DBSCAN(eps=1.0, min_samples=5),
        'Hierarchical (Single)': HierarchicalClustering(n_clusters=3, linkage='single'),
        'Hierarchical (Complete)': HierarchicalClustering(n_clusters=3, linkage='complete'),
        'GMM': GaussianMixtureModel(n_components=3)
    }
    
    results = {}
    
    print("\nTesting clustering algorithms:")
    print("-" * 40)
    
    for name, algorithm in algorithms.items():
        print(f"\nRunning {name}...")
        
        try:
            # Fit the algorithm
            algorithm.fit(X)
            
            # Get predictions
            if hasattr(algorithm, 'labels'):
                y_pred = algorithm.labels
            else:
                y_pred = algorithm.predict(X)
            
            # Handle noise points in DBSCAN
            if name == 'DBSCAN':
                n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
                n_noise = list(y_pred).count(-1)
                print(f"  Found {n_clusters} clusters and {n_noise} noise points")
            else:
                n_clusters = len(set(y_pred))
                print(f"  Found {n_clusters} clusters")
            
            # Evaluate clustering (simplified)
            if name != 'DBSCAN':  # Skip evaluation for DBSCAN due to noise points
                metrics = evaluate_clustering(y_true, y_pred)
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                results[name] = metrics['accuracy']
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = 0.0
    
    # Display results summary
    print("\n" + "=" * 40)
    print("CLUSTERING RESULTS SUMMARY")
    print("=" * 40)
    
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for i, (alg_name, accuracy) in enumerate(sorted_results):
            print(f"{i+1}. {alg_name}: {accuracy:.4f}")
    
    print("\nClustering algorithms comparison completed!")
    print("Note: Results may vary due to random initialization and data generation.")
