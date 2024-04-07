import numpy as np

def sigmoid(x):    
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def gaussian(x):
    return np.exp(-x**2)

def growing_cosine(x):
    return np.cos(x) * (1 + x)

def abs(x):
    return np.abs(x)

class NeuralNetwork:
    def __init__(self, shape=[10, 20, 4]):
        self.weights = []
        self.bias = []
        
        # Initialize weights and biases
        # [100, 10] x [10, 20], -> [100, 20]  
        # [M, N] x [N, P] = [M, P]
        
        for i in range(len(shape) - 1):
            self.weights.append(np.random.randn(shape[i], shape[i+1]) * 0.5)
            self.bias.append(np.random.randn(shape[i+1]) * 0.5)
            
    def clone(self):
        new_network = NeuralNetwork()
        new_network.weights = [w.copy() for w in self.weights]
        new_network.bias = [b.copy() for b in self.bias]
        return new_network
    
    def mse_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    def log_loss(self, predictions, targets):
        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def acc_loss(self, predictions, targets):
        # Calculate accuracy
        acc = self.acc(predictions, targets)
        
        return 1 / (acc + 1e-9)
    
    def acc(self, predictions, targets):
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(targets, axis=1)

        # Calculate accuracy
        return np.mean(pred_labels == true_labels)
         
    
    '''
    Inputs: np.array of shape (Inputs, N)
    Returns: np.array of shape (P, Outputs)
    '''
    def forward(self, inputs):
        Z = inputs
        
        for i in range(len(self.weights)-1):
            Z = np.dot(Z, self.weights[i]) + self.bias[i]
            Z = relu(Z)
        
        Z = np.dot(Z, self.weights[-1]) + self.bias[-1]  
        return sigmoid(Z)