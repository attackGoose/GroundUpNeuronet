import numpy as np
import ModelLayers.baseLayer

y_labels = np.linspace(3, 50).tolist()
x_labels = [value for value in range(0, len(y_labels))]
print(len(y_labels))

#splits 70% into training data and 30% of testing data
train_y_split = y_labels[:int(len(y_labels)*.7)] 
train_x_split = x_labels[:int(len(x_labels)*.7)]
test_y_split = y_labels[int(len(y_labels)*.7):] 
test_x_split = x_labels[int(len(x_labels)*.7):] 

model_layer_sizes = [1, 8, 4, 1]

perceptron_layer = ModelLayers.baseLayer.MultilayerPerceptron() 

multilayer_perceptron = ModelLayers.baseLayer.MultilayerPerceptron(*model_layer_sizes)
#the asterisk sign unpacks all the elements within a iterable in order into the parameters in python


epochs = 10

for epoch in range(epochs):
    perceptron_layer.backward(train_x_split, train_y_split)


print(perceptron_layer.forward(test_x_split))

