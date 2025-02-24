import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        # Initialize weights and bias
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
    
    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        # Calculate the linear combination of inputs and weights
        z = np.dot(X, self.weights) + self.bias
        # Apply sigmoid activation
        return self.sigmoid(z)
    
    def predict(self, X):
        # Get probabilities
        probas = self.predict_proba(X)
        # Convert to binary predictions using threshold of 0.5
        return (probas >= 0.5).astype(int)
    
    def train(self, X, y):
        m = len(y)  # Number of training examples
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.predict_proba(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.loss_history.append(loss)
            
            # Calculate gradients
            dw = np.dot(X.T, (y_pred - y)) / m
            db = np.sum(y_pred - y) / m
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Optional: Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()

# Generate dataset
X, y = generate_dataset()

# Train the model
neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
neuron.train(X, y)

# Plot decision boundary
plot_decision_boundary(neuron, X, y)

# Plot loss over training iterations
plot_loss(neuron)