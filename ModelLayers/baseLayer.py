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
        for x, y in zip(list(xlabel), list(ylabel)):
            error = np.subtract(ylabel, self.forward(xlabel)) 
            #sigmoid is very much needed here
            # if output = input times weight, then change in weights = error in output times input to adjust
            # since we want the size of the weight to be proportional to the size of the neuron compared to the
            # other neurons
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

        self.activation_layers = []
        self.raw_layers_values = []


        
        #creates a random weight array of size hidden layers by hidden layers, essentially all the weights needed
        #in order for our model to pass the information forward
    def backwards(self, xs, ys):
        #formula is the same as single, but this time, since multiple weights and multiple neurons
        #connect with one neuron in the next layer, we need to take the partial derivative of each
        #weight with respect to its neuron all the way back to the beginning neuron. so it will be 
        #a matrix of weights - a matrix of change in weights from the error 
        #find formula first then code it out
        formula_delta_weight: np.ndarray = lambda x, error: np.multiply(x, error)# finds the change in error

        for x, y in zip(xs, ys): #this loops over all inputs and all outputs that correspond to that input
            #standard error of output
            error = np.subtract(self.forward(x), y) 

            for layer_weights in range(0, self.weights):

                #find the error for each of the neurons of the next layer
                error_next_layer: np.ndarray  #this is the error of the next layer (vector most likely), used to adjust weights of prev layer

        #this assumes that the error that's being used for that layer would be the average of the error
            #that's been applied for each next layer, if i'm wrong please correct me
                neuron_average_error = sum(error_next_layer)/len(np.asarray(error_next_layer))
#error might be this or just the error itself applied to that layer without change w/respect to the next layer

                #this finds the new weights (which should be the transposed version of the)
                new_layer_weights = np.subtract(self.weights[layer_weights], 
                                                np.transpose(
                                                    formula_delta_weight(x=input, error=neuron_average_error)
                                                    )
                                                )
                self.weights[layer_weights] = new_layer_weights
            
            #each value should be the average change in weights of the 


    def forward(self, x: np.ndarray) -> np.ndarray:

        self.activation_layers.append(x)
        current_activation = x #start with the input, end with output

        #self.weight is an array of weight matr

        for w, b in zip(self.weights, self.biases):
            next = np.dot(w, current_activation) + b
            self.raw_layers_values.append(next)
            self.activation_layers.append(sigmoid(next))
            #next activation neuron should be the last vector of neurons that we added
            current_activation = self.activation_layers[-1] 

        return current_activation