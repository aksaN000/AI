import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict, Any


class LinearRegression:
    """
    Linear Regression implementation using gradient descent optimization.
    Supports both single and multiple variable regression.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error cost function."""
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score.
        
        Returns:
            R-squared coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegression:
    """
    Logistic Regression implementation for binary classification.
    Uses gradient descent with sigmoid activation function.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary target vector of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent optimization
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            # Compute cost (cross-entropy)
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions of shape (n_samples,)
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability predictions of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy cost function."""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class NaiveBayes:
    """
    Gaussian Naive Bayes classifier implementation.
    Assumes features follow a normal distribution within each class.
    """
    
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.priors = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Naive Bayes classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for c in self.classes:
            # Get samples belonging to class c
            X_c = X[y == c]
            
            # Calculate mean and variance for each feature
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            
            # Calculate prior probability
            self.priors[c] = len(X_c) / n_samples
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        predictions = []
        for sample in X:
            posteriors = []
            
            for c in self.classes:
                # Calculate likelihood using Gaussian probability density
                likelihood = self._gaussian_pdf(sample, self.mean[c], self.variance[c])
                
                # Calculate posterior probability (log space for numerical stability)
                posterior = np.sum(np.log(likelihood)) + np.log(self.priors[c])
                posteriors.append(posterior)
            
            # Predict class with highest posterior probability
            predictions.append(self.classes[np.argmax(posteriors)])
            
        return np.array(predictions)
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
        """Calculate Gaussian probability density function."""
        # Add small epsilon to prevent division by zero
        epsilon = 1e-9
        variance = variance + epsilon
        
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / variance)
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DecisionTree:
    """
    Decision Tree classifier implementation using CART algorithm.
    Supports both binary and multiclass classification.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the decision tree classifier.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.root = self._build_tree(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self._traverse_tree(sample, self.root) for sample in X])
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            return {'value': self._most_common_class(y)}
        
        # Find best split
        best_split = self._best_split(X, y, n_features)
        
        if best_split['info_gain'] == 0:
            return {'value': self._most_common_class(y)}
        
        # Split the data
        left_idxs = X[:, best_split['feature']] <= best_split['threshold']
        right_idxs = ~left_idxs
        
        # Check minimum samples per leaf
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            return {'value': self._most_common_class(y)}
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Dict[str, Any]:
        """Find the best split for the current node."""
        best_split = {'info_gain': 0}
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idxs = X[:, feature] <= threshold
                right_idxs = ~left_idxs
                
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue
                
                # Calculate information gain
                info_gain = self._information_gain(y, y[left_idxs], y[right_idxs])
                
                if info_gain > best_split['info_gain']:
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'info_gain': info_gain
                    }
        
        return best_split
    
    def _information_gain(self, parent: np.ndarray, left_child: np.ndarray, 
                         right_child: np.ndarray) -> float:
        """Calculate information gain using entropy."""
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Calculate weighted entropy of children
        weighted_entropy = (n_left / n_parent) * self._entropy(left_child) + \
                          (n_right / n_parent) * self._entropy(right_child)
        
        return self._entropy(parent) - weighted_entropy
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        # Calculate entropy
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _most_common_class(self, y: np.ndarray):
        """Return the most common class in the target array."""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def _traverse_tree(self, sample: np.ndarray, node: Dict[str, Any]):
        """Traverse the tree to make a prediction for a single sample."""
        if 'value' in node:
            return node['value']
        
        if sample[node['feature']] <= node['threshold']:
            return self._traverse_tree(sample, node['left'])
        else:
            return self._traverse_tree(sample, node['right'])
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Split dataset into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random permutation of indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to have zero mean and unit variance.
    
    Args:
        X: Feature matrix
        
    Returns:
        Normalized features, mean, standard deviation
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Prevent division by zero
    std[std == 0] = 1
    
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


if __name__ == "__main__":
    # Example usage and testing
    print("Machine Learning Algorithms Demo")
    print("=" * 40)
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Linear regression data
    X_reg = np.random.randn(n_samples, 1)
    y_reg = 2 * X_reg.flatten() + 1 + 0.1 * np.random.randn(n_samples)
    
    # Classification data
    X_cls = np.random.randn(n_samples, 2)
    y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)
    
    # Test Linear Regression
    print("\n1. Linear Regression")
    lr = LinearRegression(learning_rate=0.1, max_iterations=1000)
    lr.fit(X_reg, y_reg)
    r2_score = lr.score(X_reg, y_reg)
    print(f"R² Score: {r2_score:.4f}")
    print(f"Final weights: {lr.weights[0]:.4f}, bias: {lr.bias:.4f}")
    
    # Test Logistic Regression
    print("\n2. Logistic Regression")
    log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    log_reg.fit(X_cls, y_cls)
    accuracy = log_reg.accuracy(X_cls, y_cls)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Test Naive Bayes
    print("\n3. Naive Bayes")
    nb = NaiveBayes()
    nb.fit(X_cls, y_cls)
    nb_accuracy = nb.accuracy(X_cls, y_cls)
    print(f"Accuracy: {nb_accuracy:.4f}")
    
    # Test Decision Tree
    print("\n4. Decision Tree")
    dt = DecisionTree(max_depth=5)
    dt.fit(X_cls, y_cls)
    dt_accuracy = dt.accuracy(X_cls, y_cls)
    print(f"Accuracy: {dt_accuracy:.4f}")
    
    print("\nAll algorithms tested successfully!")
