from Networks.Network import DenseNetwork
from Losses.SigmoidLoss import SigmoidLoss

import numpy as np
import random

#CUSTOM TESTING DATASET
datas = []
labels = []

for i in range(0, 100):
    for j in range(0, 9):
        rand = j

        data = [float(x) for x in '{0:04b}'.format(rand)]

        label = [0.0] * 9
        label[rand] = 1.0

        datas.append(np.asarray(data).reshape((4)))
        labels.append(np.asarray(label).reshape(9))
    
datas = np.asarray(datas)
labels = np.asarray(labels)

datas_t = []
labels_t = []

for i in range(0, 200):
    rand = random.randint(0, 8)
    
    data = [float(x) for x in '{0:04b}'.format(rand)]
    
    label = [0.0] * 9
    label[rand] = 1.0
    
    datas_t.append(np.asarray(data).reshape((4)))
    labels_t.append(np.asarray(label).reshape(9))

training_data = []
testing_data = []
for d, l in zip(datas, labels):
    training_data.append((np.expand_dims(d, 1), np.expand_dims(l, 1)))
    
for d, l in zip(datas_t, labels_t):
    testing_data.append((np.expand_dims(d, 1), np.expand_dims(l, 1)))

training_data = np.array(training_data)
testing_data = np.array(testing_data)

print(np.shape(training_data[0][0]))
print(np.shape(training_data[0][1]))

network = DenseNetwork([4, 9], SigmoidLoss(), 1)
network.compile()

result = network.feedforward(training_data[0][0])
print(result)

res = network.train(training_data, testing_data, 100, 0.1, 0)
#network.saveToFile('losses_test')