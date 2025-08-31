import numpy as np
import math
from typing import List, Tuple, Callable, Optional


class NeuralNetwork:
    """
    A flexible multi-layer neural network implementation from scratch.
    Supports customizable architectures, activation functions, and optimizers.
    """
    
    def __init__(self, layers: List[int], activation: str = 'relu', 
                 output_activation: str = 'sigmoid', learning_rate: float = 0.01):
        """
        Initialize neural network.
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            output_activation: Activation function for output layer ('sigmoid', 'softmax', 'linear')
            learning_rate: Learning rate for gradient descent
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases using Xavier initialization
        for i in range(len(layers) - 1):
            # Xavier initialization for better convergence
            limit = math.sqrt(6.0 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            b = np.zeros((1, layers[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation functions
        self.activation = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
        self.output_activation = self._get_activation_function(output_activation)
        self.output_activation_derivative = self._get_activation_derivative(output_activation)
        
        # Training history
        self.loss_history = []
        
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            output: Network output
            activations: List of layer activations
            z_values: List of pre-activation values
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.activation(z)
            activations.append(a)
            current_input = a
        
        # Output layer
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        output = self.output_activation(z_output)
        activations.append(output)
        
        return output, activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                z_values: List[np.ndarray]) -> None:
        """
        Backward propagation to compute gradients and update weights.
        
        Args:
            X: Input data
            y: True labels
            activations: List of layer activations from forward pass
            z_values: List of pre-activation values from forward pass
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.output_activation == self._sigmoid:
            # For sigmoid output (binary classification)
            delta = activations[-1] - y
        else:
            # For other activations, use derivative
            delta = (activations[-1] - y) * self.output_activation_derivative(z_values[-1])
        
        # Gradients for output layer
        dW[-1] = np.dot(activations[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Propagate error to previous layer
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(z_values[i])
            
            # Compute gradients
            dW[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            batch_size: Optional[int] = None, verbose: bool = True) -> None:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples, n_outputs)
            epochs: Number of training epochs
            batch_size: Size of mini-batches (None for full batch)
            verbose: Whether to print training progress
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if batch_size is None:
            batch_size = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                output, activations, z_values = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations, z_values)
                
                # Calculate loss
                batch_loss = self._compute_loss(y_batch, output)
                epoch_loss += batch_loss
                num_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions
        """
        output, _, _ = self.forward(X)
        
        # For binary classification with sigmoid
        if output.shape[1] == 1:
            return (output > 0.5).astype(int)
        # For multi-class classification
        else:
            return np.argmax(output, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Class probabilities
        """
        output, _, _ = self.forward(X)
        return output
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss (binary cross-entropy or mean squared error)."""
        if y_true.shape[1] == 1 and self.output_activation == self._sigmoid:
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean squared error
            return np.mean((y_true - y_pred) ** 2)
    
    def _get_activation_function(self, name: str) -> Callable:
        """Get activation function by name."""
        if name == 'relu':
            return self._relu
        elif name == 'sigmoid':
            return self._sigmoid
        elif name == 'tanh':
            return self._tanh
        elif name == 'softmax':
            return self._softmax
        elif name == 'linear':
            return self._linear
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    def _get_activation_derivative(self, name: str) -> Callable:
        """Get activation function derivative by name."""
        if name == 'relu':
            return self._relu_derivative
        elif name == 'sigmoid':
            return self._sigmoid_derivative
        elif name == 'tanh':
            return self._tanh_derivative
        elif name == 'softmax':
            return self._softmax_derivative
        elif name == 'linear':
            return self._linear_derivative
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    # Activation functions and their derivatives
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _softmax_derivative(self, x):
        # Simplified for backpropagation
        return np.ones_like(x)
    
    def _linear(self, x):
        return x
    
    def _linear_derivative(self, x):
        return np.ones_like(x)


class Perceptron:
    """
    Single-layer perceptron for binary classification.
    Classic linear classifier with step function activation.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the perceptron using the perceptron learning rule.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Binary labels of shape (n_samples,) with values in {0, 1} or {-1, 1}
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Convert labels to {-1, 1} if they are {0, 1}
        if np.all(np.isin(y, [0, 1])):
            y = 2 * y - 1
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(self.max_iterations):
            errors = 0
            
            for i in range(n_samples):
                # Make prediction
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self._step_function(linear_output)
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    errors += 1
            
            # If no errors, the data is linearly separable and we're done
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Binary predictions {-1, 1}
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self._step_function(x) for x in linear_output])
    
    def _step_function(self, x: float) -> int:
        """Step activation function."""
        return 1 if x >= 0 else -1
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        # Convert labels to {-1, 1} if they are {0, 1}
        if np.all(np.isin(y, [0, 1])):
            y = 2 * y - 1
        
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RBFNetwork:
    """
    Radial Basis Function (RBF) Neural Network.
    Uses Gaussian RBF functions in the hidden layer.
    """
    
    def __init__(self, n_hidden: int = 10, gamma: float = 1.0, learning_rate: float = 0.01):
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.centers = None
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> None:
        """
        Train the RBF network.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            epochs: Number of training epochs
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize RBF centers using k-means-like approach
        self.centers = self._initialize_centers(X)
        
        # Initialize output weights
        self.weights = np.random.randn(self.n_hidden, y.shape[1]) * 0.1
        self.bias = np.zeros((1, y.shape[1]))
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            rbf_output = self._rbf_layer(X)
            predictions = np.dot(rbf_output, self.weights) + self.bias
            
            # Compute error
            error = y - predictions
            
            # Update output weights (only output layer is trained)
            self.weights += self.learning_rate * np.dot(rbf_output.T, error) / n_samples
            self.bias += self.learning_rate * np.mean(error, axis=0, keepdims=True)
            
            if epoch % 100 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}, MSE: {mse:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predictions
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        rbf_output = self._rbf_layer(X)
        predictions = np.dot(rbf_output, self.weights) + self.bias
        
        # For binary classification
        if predictions.shape[1] == 1:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions
    
    def _initialize_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialize RBF centers using random selection from training data."""
        indices = np.random.choice(X.shape[0], self.n_hidden, replace=False)
        return X[indices].copy()
    
    def _rbf_layer(self, X: np.ndarray) -> np.ndarray:
        """
        Compute RBF layer output using Gaussian functions.
        
        Args:
            X: Input data
            
        Returns:
            RBF layer activations
        """
        n_samples = X.shape[0]
        rbf_output = np.zeros((n_samples, self.n_hidden))
        
        for i, center in enumerate(self.centers):
            # Compute squared Euclidean distance
            distances_sq = np.sum((X - center) ** 2, axis=1)
            # Apply Gaussian RBF function
            rbf_output[:, i] = np.exp(-self.gamma * distances_sq)
        
        return rbf_output


def create_spiral_dataset(n_samples: int = 100, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a two-spiral dataset for testing neural networks.
    
    Args:
        n_samples: Number of samples per spiral
        noise: Amount of noise to add
        
    Returns:
        X: Input features, y: Binary labels
    """
    n = n_samples // 2
    
    # Generate first spiral
    theta1 = np.linspace(0, 4 * np.pi, n)
    r1 = np.linspace(0.1, 1, n)
    x1 = r1 * np.cos(theta1) + noise * np.random.randn(n)
    y1 = r1 * np.sin(theta1) + noise * np.random.randn(n)
    spiral1 = np.column_stack([x1, y1])
    labels1 = np.zeros(n)
    
    # Generate second spiral
    theta2 = np.linspace(0, 4 * np.pi, n) + np.pi
    r2 = np.linspace(0.1, 1, n)
    x2 = r2 * np.cos(theta2) + noise * np.random.randn(n)
    y2 = r2 * np.sin(theta2) + noise * np.random.randn(n)
    spiral2 = np.column_stack([x2, y2])
    labels2 = np.ones(n)
    
    # Combine and shuffle
    X = np.vstack([spiral1, spiral2])
    y = np.hstack([labels1, labels2])
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


if __name__ == "__main__":
    # Example usage and demonstrations
    print("Neural Network Implementations Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create test datasets
    print("\nGenerating test datasets...")
    
    # XOR dataset for testing non-linear separation
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # Spiral dataset
    X_spiral, y_spiral = create_spiral_dataset(200, noise=0.05)
    
    # Test Multi-layer Neural Network on XOR
    print("\n1. Multi-layer Neural Network (XOR Problem)")
    nn = NeuralNetwork([2, 4, 1], activation='relu', learning_rate=0.1)
    nn.fit(X_xor, y_xor, epochs=2000, verbose=False)
    
    predictions = nn.predict(X_xor)
    accuracy = np.mean(predictions.flatten() == y_xor)
    print(f"XOR Accuracy: {accuracy:.4f}")
    print("XOR Predictions:", predictions.flatten())
    print("XOR True labels:", y_xor)
    
    # Test Perceptron on linearly separable data
    print("\n2. Perceptron (Linearly Separable Data)")
    # Create linearly separable dataset
    X_linear = np.random.randn(100, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
    
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.fit(X_linear, y_linear)
    perceptron_accuracy = perceptron.accuracy(X_linear, y_linear)
    print(f"Perceptron Accuracy: {perceptron_accuracy:.4f}")
    
    # Test RBF Network on spiral data
    print("\n3. RBF Network (Spiral Dataset)")
    rbf = RBFNetwork(n_hidden=15, gamma=2.0, learning_rate=0.01)
    rbf.fit(X_spiral, y_spiral, epochs=500)
    
    rbf_predictions = rbf.predict(X_spiral)
    rbf_accuracy = np.mean(rbf_predictions == y_spiral)
    print(f"RBF Network Accuracy: {rbf_accuracy:.4f}")
    
    # Test deep network on spiral data
    print("\n4. Deep Neural Network (Spiral Dataset)")
    deep_nn = NeuralNetwork([2, 10, 8, 6, 1], activation='tanh', learning_rate=0.01)
    deep_nn.fit(X_spiral, y_spiral, epochs=1000, verbose=False)
    
    deep_predictions = deep_nn.predict(X_spiral)
    deep_accuracy = np.mean(deep_predictions.flatten() == y_spiral)
    print(f"Deep Network Accuracy: {deep_accuracy:.4f}")
    
    print("\nAll neural network models tested successfully!")
