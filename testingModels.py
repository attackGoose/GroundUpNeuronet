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

model_layer_sizes = [1, 4, 1]
print(*model_layer_sizes)
multilayer_perceptron = ModelLayers.baseLayer.MultilayerPerceptron(model_layer_sizes)
#the asterisk sign unpacks all the elements within a iterable in order into the parameters in python


epochs = 10

for epoch in range(epochs):
    multilayer_perceptron.backwards(train_x_split, train_y_split)


print(multilayer_perceptron.forward(test_x_split))


#too little data would usually result in over fitting when it comes to more diverse datasets
#so I'll prob get a lot more data for the tansformer models