import numpy as np

#I'm not sure how to calculate the bias term, Idk if its something it figures out by itself or smth
#usually its derivative is equal to 1, but it still has a slight affect on the error, so wut i do with it


def sigmoid(input: float) -> float: #1/(1+e^-x)
    return 1/(1+np.exp(-input))

def sigmoid_back(x: int) -> float:
  fwd = sigmoid(x)
  return fwd * (1-fwd)

class Perceptron: #also figure out if I can do math stuff as well
    #test out what bias does
    def __init__(self, input_shape, bias = 0, activation: function = sigmoid) -> None:
        self.input = input
        #series of weights
        self.weights = np.random.rand()
        #series of biases
        self.bias = np.multiply(bias, np.ones(self.weights.shape))
        self.activation_func = activation

    
    def backward(self, xlabel, ylabel, layerGradient = None, learningRate = None):
        #get the error, not sure if activation function (sigmoid) is going to affect the result or not,
        #trial and error
        error = np.subtract(ylabel, self.forward(xlabel))
        #error in terms of y label, and ylabel = weight times x, so error times input should give you the change,
        # its the other side of the coin, 
        # if output = input times weight, then change in weights = error in output times input, 
        # intuitively makes sense
        change_in_weight = np.multiply(error, xlabel)
        self.weights = np.subtract(self.weights, change_in_weight)

    def forward(self, input):
        return self.activation_func(input*self.weights) + self.bias





#next step is to implement a model with multiple neurons and weights

class MultilayerPerceptron:
    def __init__(self, array_sizes: list): 
        self.input_size = np.random.rand(array_sizes[1])
        self.output_size = np.random.rand(array_sizes[-1])
        #i'm making this a array of 2d arrays to store all the weights of all the connections between neurons
        #the sizes will be the output:input
        self.weights = [np.random.rand(y, x) for y, x in zip(array_sizes[1:], array_sizes[:-1])]
        self.bias = [np.random.rand(x) for x in array_sizes[1:-1]]

        self.sizes = array_sizes
        self.num_layers = len(array_sizes)


        
        #creates a random weight array of size hidden layers by hidden layers, essentially all the weights needed
        #in order for our model to pass the information forward
    def backwards(self, xs, ys):
        #find formula first then code it out
        for x, y in zip(xs, ys):
            error = np.subtract(self.forward(x), y)

            
    def forward(self, x: np.ndarray) -> np.ndarray:

        activation_layers = [x]
        current_activation = x #start with the input
        raw_layers_values = []

        #self.weight is an array of weight matr

        for w, b in zip(self.weights, self.biases):
            next = np.dot(w, activation) + b
            raw_layers_values.append(next)
            activation_layers.append(sigmoid(next))
            activation = activation_layers[-1] #next activation neuron should be the last vector of neurons that we added
