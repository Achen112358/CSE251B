################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2022
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import pickle
import random
import time
from sklearn.metrics import confusion_matrix

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp, mu=None, sigma=None):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    inp = np.array(inp.copy())
    n, m = inp.shape
    inp = inp.reshape(n, 3, 1024).transpose(1,0,2).reshape(3, -1)
    
    if mu is None:
        mu = np.mean(inp, axis=-1, keepdims=True)
    if sigma is None:
        sigma = np.std(inp, axis=-1, keepdims=True)
        
    inp = (inp - mu) / sigma
    inp = inp.reshape(3, n, 1024).transpose(1,0,2).reshape(n, m)
    
    return inp, mu, sigma


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    KI = np.eye(num_classes)
    labels = KI[labels]
    return labels

def shuffle(dataset):
    X, y = dataset
    n = len(X)
    indexes = list(range(n))
    random.shuffle(indexes)
    return (X[indexes], y[indexes])

def generate_minibatches(dataset, batch_size=64):
    X, y = shuffle(dataset)
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def load_data(path):
    """
    Load CIFAR-10 data.
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, "cifar-10-batches-py")
    
    images = []
    labels = []
    for i in range(1,6):
        images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
        data = images_dict[b'data']
        label = images_dict[b'labels']
        labels.extend(label)
        images.extend(data)
    normalized_images_train, mu, sigma = normalize_data(images)
    one_hot_labels_train    = one_hot_encoding(labels, num_classes=10) #(n,10)

    test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'labels']
    normalized_images_test, _, __ = normalize_data(test_data, mu, sigma)
    one_hot_labels_test    = one_hot_encoding(test_labels, num_classes=10) #(n,10)
    return  np.array(normalized_images_train), np.array(one_hot_labels_train), \
            np.array(normalized_images_test), np.array(one_hot_labels_test)
    


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    x = np.array(x)
    x -= np.max(x, axis=-1, keepdims=True) #in case exp(x) is too large to be nan
    return np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims=True)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()
            
        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        x = np.maximum(-700, x)
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x) 
        
    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0, x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        
        return np.maximum(x*0.1, x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        x = np.maximum(-700, self.x)
        val = self.sigmoid(x)
        return  val * (1.0 - val) 

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        
        return 1 - self.tanh(self.x)**2

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        grad = self.x.copy()
        grad[grad>0] = 1
        grad[grad<0] = 0
        return grad

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        grad = self.x.copy()
        grad[grad>0] = 1
        grad[grad<=0] = 0.1
        return grad


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)   # Declare the Weight matrix
        self.b = np.random.randn(1, out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        
        self.d_w_old = 0
        self.d_b_old = 0
        self.w_best = None
        self.b_best= None

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = x.copy()
        self.a = np.matmul(x, self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        batch_size = self.x.shape[0]

        self.d_x = np.matmul(delta, self.w.T)
        self.d_w = -np.matmul(self.x.T, delta) / (batch_size * 10)
        self.d_b = -np.mean(delta, axis=0) / 10

        return self.d_x
    
    def update(self, lr, l2_penalty = 0, momentum = False, momentum_gamma = 0.9):
        
        # l2 penalty
        d_w  = self.d_w + l2_penalty * self.w
        
        #momentum
        if momentum:
            d_w = (1-momentum_gamma) * d_w +  momentum_gamma * self.d_w_old 
            d_b = (1-momentum_gamma) * self.d_b + momentum_gamma * self.d_b_old
            self.d_w_old = d_w
            self.d_b_old = d_b
        
        self.w -= lr * d_w
        self.b -= lr * d_b
    
    def store(self):
        self.w_best = self.w
        self.b_best = self.b

    def load(self):
        self.w = self.w_best
        self.b = self.b_best
        

class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.l2_penalty = config['L2_penalty']
        self.lr = config['learning_rate']
        self.momentum = config['momentum']
        self.momentum_gamma = config['momentum_gamma']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x.copy()
        self.targets = targets
        
        inp = x.copy()
        for layer in self.layers:
            inp = layer.forward(inp)
            
        self.y = softmax(inp)
        
        if targets is not None:
            loss = self.loss(self.y, self.targets)
            return self.y, loss
        else:
            return self.y
            

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        loss = - np.mean(np.sum(np.log(logits + 1e-9) * targets, axis=-1))
        return loss
        

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = -(self.y - self.targets)
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
            
    def update(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.update(lr=self.lr, l2_penalty = self.l2_penalty, momentum = self.momentum, momentum_gamma = self.momentum_gamma)
                
    def store(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.store()

    def load(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.load()
                
    def predict_acc(self, x, targets):
        y = self.forward(x)
        predictions = np.argmax(y, axis=1)
        targets = np.argmax(targets, axis=1)
        return np.mean(predictions == targets)
    
    def predict(self, x, targets):
        y = self.forward(x)
        predictions = np.argmax(y, axis=1)
        return predictions


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    
    start_time = time.time()
        
    best_loss = 1000.0
    patient = 0
    final_train_loss, final_train_acc, final_valid_loss, final_valid_acc = [], [], [], []
    for t in range(config['epochs']):
        train_loss_batch, train_accuracy_batch = [], []
        for x, y in generate_minibatches((x_train, y_train), batch_size=config['batch_size']):
            opt, loss = model.forward(x, targets=y)
            train_loss_batch.append(loss)
            model.backward()
            model.update()   
            acc = model.predict_acc(x, y)
            train_accuracy_batch.append(acc)
            
        train_loss = np.mean(np.array(train_loss_batch))
        train_accuracy = np.mean(np.array(train_accuracy_batch))
        final_train_loss.append(train_loss)
        final_train_acc.append(train_accuracy)
        
        print("Epoch {}: Time cost {}s. Training loss is {}. Training Accuracy is {}.".format(t + 1, round(time.time() - start_time, 2), round(train_loss,4), round(train_accuracy, 4)))
        
        valid_loss = model.forward(x_valid, targets=y_valid)[1]
        valid_acc = model.predict_acc(x_valid, targets=y_valid)
        final_valid_loss.append(valid_loss)
        final_valid_acc.append(valid_acc)
        
        print('Begin Validation! Time cost {}s. Validation loss is {}. Validation Accuracy is {}.'.format(round(time.time() - start_time, 2), round(valid_loss,4), round(valid_acc, 4)))
        if valid_loss < best_loss:
            model.store()
            best_loss = valid_loss
            patient = 0
        else:
            patient += 1
        if patient > config['early_stop_epoch']:
            break
            
    return final_train_loss, final_train_acc, final_valid_loss, final_valid_acc
    
def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    acc = model.predict_acc(X_test, targets=y_test)
    return acc



if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    path = "./data/"
    X, y, X_test, y_test = load_data(path)
    X, y = shuffle((X,y))
    X_train, X_valid = X[:45000], X[45000:]
    y_train, y_valid = y[:45000], y[45000:]

    # TODO: train the model
    train(model, X_train, y_train, X_valid, y_valid, config)
    test_acc = test(model, X_test, y_test)

    # TODO: Plots
    # plt.plot(...)
