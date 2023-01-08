from Networks.Dense import DenseNetwork
from Activations.SigmoidActivation import SigmoidActivation
from Losses.MeanSquaredError import MeanSquaredError

import numpy as np
import random
import pickle

#CUSTOM TESTING DATASET
# datas = []
# labels = []

# for i in range(0, 100):
#     for j in range(0, 9):
#         rand = j

#         data = [float(x) for x in '{0:04b}'.format(rand)]

#         label = [0.0] * 9
#         label[rand] = 1.0

#         datas.append(np.asarray(data).reshape((4)))
#         labels.append(np.asarray(label).reshape(9))
    
# datas = np.asarray(datas)
# labels = np.asarray(labels)

# datas_t = []
# labels_t = []

# for i in range(0, 200):
#     rand = random.randint(0, 8)
    
#     data = [float(x) for x in '{0:04b}'.format(rand)]
    
#     label = [0.0] * 9
#     label[rand] = 1.0
    
#     datas_t.append(np.asarray(data).reshape((4)))
#     labels_t.append(np.asarray(label).reshape(9))

data = pickle.load(open('./Data/BTCUSDT.data', 'rb'))

training_data = []
testing_data = []
for d in data[:1501]:
    training_data.append((np.expand_dims(d[0], 1), np.expand_dims(d[1], 1)))
    
for d in data[1501:]:
    testing_data.append((np.expand_dims(d[0], 1), np.expand_dims(d[1], 1)))

training_data = np.array(training_data)
testing_data = np.array(testing_data)

print(np.shape(training_data[0][0]))
print(np.shape(training_data[0][1]))

network = DenseNetwork([21, 10, 10, 1], MeanSquaredError(), SigmoidActivation())
network.compile()

res = network.train(training_data, testing_data, 30, 0.001, 0)
training_hist_file = open('./Output/training_history.csv', 'w')
training_hist_file.write('train_loss;train_acc;val_loss;val_acc\n')
for epoch in res:
    training_hist_file.write(f'{epoch[0][0]};{epoch[0][1]};{epoch[1][0]};{epoch[1][1]}\n')
#network.saveToFile('losses_test')

