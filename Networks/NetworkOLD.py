import numpy as np
import random
import pickle
import queue
from WeightsVisualisation import visualise
from threading import Thread

class DenseNetworkOld:
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

        weights = []
        biases = []

        for i in range(1, len(self.sizes)):
            weights.append(np.random.randn(self.sizes[i], self.sizes[i - 1]))
            #weights.append(np.zeros((self.sizes[i], self.sizes[i - 1])))
            biases.append(np.random.randn(self.sizes[i], 1))

        self.weights = np.asarray(weights)
        self.biases = np.asarray(biases)

    def feedforward(self, x):
        activation = x
        for weight_l, bias_l in zip(self.weights, self.biases):
            #Iterate layers
            temp_activation = []
            for weight_n, bias_n in zip(weight_l, bias_l):
                #Iterate neurons
                temp_activation.append(sigmoid(np.sum(activation * weight_n) + bias_n).flatten())
            activation = np.asarray(temp_activation).flatten()
    
        return activation

    def evaluate(self, testing_dataset):
        cost = 0
        match = 0
        for x, y in testing_dataset:
            prediction = self.feedforward(x)
            difference = (prediction - y)
            cost += (np.sqrt(np.dot(difference, difference)) ** 2) / len(testing_dataset) #MSE loss function
            if (np.argmax(prediction) == np.argmax(y)):
                match += 1
        accuracy = match / len(testing_dataset)
    
        return (cost, accuracy)

    def train(self, training_data, testing_data, epochs = 10, learning_rate = 0.01, enable_gfx = False, text_queue = None, statistics_list = None):
        #GUI variables
        self.training_statistics = statistics_list
        self.training_output = text_queue

        #Visualisation variables
        graphics_thread = None
        graphics_thread_run = [False]
        visualisation_weights = []
        
        #PyGame visualisation    
        if enable_gfx:
            for i in range(0, self.weights.shape[0]):
                visualisation_weights.append(np.copy(self.weights[i]))
            graphics_thread_run[0] = True
            graphics_thread = Thread(target = visualise, args = (self.sizes, visualisation_weights, graphics_thread_run))
            graphics_thread.start()
        
        evaluation = []
        epochs = epochs
        learning_rate = learning_rate
        for i in range(0, epochs):
            print('Epoch [{}/{}]'.format(i + 1, epochs))
        
            #Randomly shuffle training dataset
            np.random.shuffle(training_data)
        
            counter = 0
            for x, y in training_data:
                #Batch size = 1 HARDCODED
                counter += 1
                if (counter == 50):
                    e_res = self.evaluate(training_data)
                    print('\rtrain_loss: {:.5f} train_acc: {:.2f}'.format(e_res[0], e_res[1]), end = '')
                    counter = 0
                
                #PyGame visualisation    
                if enable_gfx: 
                    for k in range(0, self.weights.shape[0]):
                        visualisation_weights[k] = np.copy(self.weights[k])

            
                #FEEDFORWARD - compute activations and pre_activations (a, z) for each layer
                activation = x
            
                activations = [x]
                pre_activations = [x] #Might be a mistake to add x
            
                for weight_l, bias_l in zip(self.weights, self.biases):
                    #Iterate layers
                    temp_preactivation = []
                    temp_activation = []
                    for weight_n, bias_n in zip(weight_l, bias_l):
                        #Iterate neurons
                        pre_activation = (np.sum(activation * weight_n) + bias_n).flatten()
                        temp_preactivation.append(pre_activation) #z
                        temp_activation.append(sigmoid(pre_activation)) #a
                    pre_activation = np.asarray(temp_preactivation).flatten()
                    activation = np.asarray(temp_activation).flatten()
                
                    pre_activations.append(pre_activation)
                    activations.append(activation)
                
                #CALCULATE DELTA FOR WEIGHTS AND BIASES
                delta_w = [np.zeros_like(x) for x in self.weights]
                delta_b = [np.zeros_like(x) for x in self.biases]
            
                #Calculate error gradient for all weights
                error_gradient = [None] * (self.weights.shape[0])
                error_gradient[self.weights.shape[0] - 1] = (activations[self.weights.shape[0]] - y) * sigmoid_prime(pre_activations[self.weights.shape[0]])
                #print('{} - {} * {}'.format(activations[weights.shape[0]], y, sigmoid_prime(pre_activations[weights.shape[0]]))) #DEBUG
                #Backpropagate error to calculate all gradients along the way
                for l in range(self.weights.shape[0] - 2, -1, -1): #0 to 0
                    layer_error = []
                    for j in range(0, len(pre_activations[l + 1])):
                        #temp = 0
                        weights_t = np.transpose(self.weights[l + 1], (1, 0))
                        temp = weights_t[j] * error_gradient[l + 1] * sigmoid_prime(pre_activations[l + 1][j])
                        temp = np.sum(temp, axis = 0)
                        ##Old, unoptimised
                        #for k in range(0, len(error_gradient[l + 1])):
                        #    #temp += self.weights[l + 1][k][j] * error_gradient[l + 1][k] * sigmoid_prime(pre_activations[l + 1][j])
                        #    temp += weights_t[j][k] * error_gradient[l + 1][k] * sigmoid_prime(pre_activations[l + 1][j])
                        layer_error.append(temp)
                    error_gradient[l] = layer_error
            
                #Gradient of cost function with respect to self.weights
                for l in range(self.weights.shape[0] - 1, -1, -1): #1 to 0
                    for k in range(0, delta_w[l].shape[0]):
                        d_weight_n = delta_w[l][k]
                        error = error_gradient[l][k]
                        
                        delta_w[l][k] = activations[l] * error
                        #Old, unoptimised
                        #for j in range(0, d_weight_n.shape[0]):
                        #    d_weight_n[j] = activations[l][j] * error
            

                #Gradient of cost function with respect to biases
                for l in range(self.biases.shape[0] - 1, -1, -1): #1 to 0
                    for j in range(0, self.biases[l].shape[0]):
                        delta_b[l][j] = error_gradient[l][j]

                delta_w = np.asarray(delta_w)
                                
                #UPDATE WEIGHTS AND BIASES
                self.weights = np.asarray([w - learning_rate * dw for w, dw in zip(self.weights, delta_w)])
                self.biases = np.asarray([b - learning_rate * db for b, db in zip(self.biases, delta_b)])
            
            e_res = self.evaluate(testing_data)
            evaluation.append(e_res)
            print(' val_loss: {:.5f} val_acc: {:.2f}'.format(e_res[0], e_res[1]), end = '') #Console
            
            #GUI text output
            if (self.training_output is not None):
                self.training_output.put('val_loss: {:.5f} val_acc: {:.2f}\n'.format(e_res[0], e_res[1]))
                self.training_output.put('Epoch [{}/{}]\n'.format(i + 1, epochs))
            #GUI matplotlib statistics
            if (self.training_statistics is not None):
                self.training_statistics.append((i + 1, e_res))

            print('')
    
        if enable_gfx:
            graphics_thread_run[0] = False
            graphics_thread.join()
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