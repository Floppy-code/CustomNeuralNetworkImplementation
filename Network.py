import numpy as np
import pickle
from threading import Thread
import random

class Network:
    """An implementation of Neural Network based on SGD with sigmoid activation function"""

    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = []
        self.biases = []

        self.training_statistics = None #List of validation loss values along with epoch number.
        self.training_output = None     #Console training output queue.

        self.isCompiled = False


    def compile(self):
        self.isCompiled = True

        [print(f'{x} | {y}') for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]


    def feedforward(self, x):
        for weight_l, bias_l in zip(self.weights, self.biases):
            x = sigmoid(np.dot(weight_l, x) + bias_l)
    
        return x

    #Returns delta weights and delta biases to be used for optimization of current values.
    def backpropagate(self, x, y):
        #FEEDFORWARD - compute activations and pre_activations (a, z) for each layer
        activation = x
        activations = [x]
        pre_activations = []
    
        #CALCULATE DELTA FOR WEIGHTS AND BIASES
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        for weight_l, bias_l in zip(self.weights, self.biases):
            z = np.dot(weight_l, activation) + bias_l
            pre_activations.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #Calculate error gradient for last layer of neurons
        error_gradient = (activations[-1] - y) * sigmoid_prime(pre_activations[-1])
        delta_w[-1] = error_gradient @ activations[-2].T
        delta_b[-1] = error_gradient

        for l in range(2, len(self.sizes)):
            z = pre_activations[-l]
            sp = sigmoid_prime(z)
            error_gradient = np.dot(self.weights[-l + 1].transpose(), error_gradient) * sp
            delta_w[-l] = np.dot(error_gradient, activations[-l - 1].transpose())
            delta_b[-l] = error_gradient

        return (delta_w, delta_b)
    
    #Update current weights a biases.
    def update_batch(self, delta_w, delta_b, lr):
        self.weights = [w - dw * lr for w, dw in zip(self.weights, delta_w)]
        self.biases = [b - db * lr for b, db in zip(self.biases, delta_b)]
            

    def evaluate(self, testing_dataset):
        cost = 0
        match = 0
        for x, y in testing_dataset:
            prediction = self.feedforward(x)
            difference = (prediction - y).flatten()
            cost += (np.sqrt(np.dot(difference, difference)) ** 2) / len(testing_dataset) #MSE loss function
            if (np.argmax(prediction) == np.argmax(y)):
                match += 1
        accuracy = match / len(testing_dataset)
    
        return (cost, accuracy)


    def train(self, training_data, testing_data, epochs = 10, learning_rate = 0.01, verbose = 0):             
        evaluation = []
        for i in range(0, epochs):
            print('Epoch [{}/{}]'.format(i + 1, epochs))
        
            #Randomly shuffle training dataset
            np.random.shuffle(training_data)
        
            #One iteration over all data
            counter = 0
            for x, y in training_data:
                #Batch size = 1 HARDCODED
                counter += 1
                if (counter == 50):
                    e_res = self.evaluate(training_data)
                    print('\rtrain_loss: {:.5f} train_acc: {:.2f}'.format(e_res[0], e_res[1]), end = '')
                    counter = 0
            
                #Apply backpropagation to current weights and biases
                delta_w, delta_b = self.backpropagate(x, y)
                self.update_batch(delta_w, delta_b, learning_rate)
                
            
            e_res = self.evaluate(testing_data)
            evaluation.append(e_res)
            print(' val_loss: {:.5f} val_acc: {:.2f}'.format(e_res[0], e_res[1]), end = '') #Console
            
            #GUI text output
            if (verbose == 1):
                print('Epoch [{}/{}]\n'.format(i + 1, epochs))
                print('val_loss: {:.5f} val_acc: {:.2f}\n'.format(e_res[0], e_res[1]))

            print('')
        return evaluation

    #TODO
    def saveToFile(self, filepath):
        buffer = (self.sizes, self.weights, self.biases, self.isCompiled, self.cost)
        file = open(filepath, 'wb')
        pickle.dump(buffer, file)
        file.close()

    #TODO
    def loadFromFile(self, filepath):
        file = open(filepath, 'rb')
        input = pickle.load(file)
        self.sizes, self.weights, self.biases, self.isCompiled, self.cost = input
        file.close()

#MISC
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))