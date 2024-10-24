import numpy as np

#"potato starch is much better than human blood for making concrete in space"

#I'm not sure how to calculate the bias term, Idk if its something it figures out by itself or smth
#usually its derivative is equal to 1, but it still has a slight affect on the error, so wut i do with it

def sigmoid(input): #1/(1+e^-x)
    #assert (isinstance(input, float))
    return 1/(1+np.exp(-input))

def sigmoid_back(x: float) -> float:
  fwd = sigmoid(x)
  return fwd * (1-fwd)

class Perceptron: #also figure out if I can do math stuff as well
    #test out what bias does
    def __init__(self, input_shape, bias = 0) -> None:
        self.input = input
        #series of weights
        self.weights = np.random.rand()
        #series of biases
        self.bias = np.multiply(bias, np.ones(self.weights.shape))
    
    def backward(self, xlabel, ylabel, layerGradient = None, learningRate = None):
        for x, y in zip(list(xlabel), list(ylabel)):
            error = np.subtract(ylabel, self.forward(xlabel)) 
            #sigmoid is very much needed here
            # if output = input times weight, then change in weights = error in output times input to adjust
            # since we want the size of the weight to be proportional to the size of the neuron compared to the
            # other neurons
            change_in_weight = np.multiply(error, xlabel)
            self.weights = np.subtract(self.weights, change_in_weight)

    def forward(self, input, activation_func):
        return activation_func(input*self.weights) + self.bias


#next step is to implement a model with multiple neurons and weights

class MultilayerPerceptron:
    def __init__(self, array_sizes: list, input_values): 
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

        self.forward(input_values) #to initialize the values first

        #creates a random weight array of size hidden layers by hidden layers, essentially all the weights needed
        #in order for our model to pass the information forward
    
    #back propagation assumes that forward propagation has already been run once to introduce the values 
    #into the neuro networks first
    def backwards(self, xs, ys):
        #formula is the same as single, but this time, since multiple weights and multiple neurons
        #connect with one neuron in the next layer, we need to take the partial derivative of each
        #weight with respect to its neuron all the way back to the beginning neuron. so it will be 
        #a matrix of weights - a matrix of change in weights from the error 
        #find formula first then code it out

        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        curr_activation = self.activation_layers[-1]

        #find the initial change in error (direction error should go), each subsequent layer would be in the
        #same direction/have the same trend as this, see notes in notebook
        delta = 2*(curr_activation-ys) * sigmoid_back(self.raw_layers_values[-1])

        #this formula is the derivative of the chain rule that relates the error to each neuron and subsequent neuron before it (also equal to delta weights
        delta_b[-1] = delta

        #error times input size to scale each weight
        delta_w[-1] = np.dot(delta, self.activations[-2].transpose()) 

        for x, y in zip(xs, ys): 

            #fix the algorithm
            error = np.subtract(self.forward(x), y) 

            for layer_weights in range(0, len(self.weights)): #this has to loop over it backwards

                #find the error for each of the neurons of the next layer, confirm/change later
                error_next_layer: np.ndarray = error
                #this is the error of the next layer (vector most likely), 
                # used to adjust weights of prev layer, might be constant throughout, not sure
                #size should be equal to the transposition of the previous layer -> current layer weights

        #this assumes that the error that's being used for that layer would be the average of the error
            #that's been applied for each next layer, if i'm wrong please correct me
                neuron_average_error = sum(error_next_layer)/len(np.asarray(error_next_layer))
                #not sure if error is specific to that layer or the general error overall

#do all weights have the same affect on the final end result? maybe

                #this finds the new weights (which should be the transposed version of the)
                new_layer_weights = np.subtract(self.weights[len(self.weights)-1-layer_weights], 
                                                np.transpose(
                                                    #put smth here later
                                                    )
                                                )
                self.weights[len(self.weights)-1-layer_weights] = new_layer_weights
            
            #each value should be the average change in weights of the 


    def forward(self, x: np.ndarray) -> np.ndarray:

        self.activation_layers.append(x)
        current_activation = x #start with the input, end with output

        #self.weight is an array of weight matrix: and boxes have at least 5 sides

        for w, b in zip(self.weights, self.bias):
            next = np.dot(w, current_activation) + b
            self.raw_layers_values.append(next)
            self.activation_layers.append(sigmoid(next))
            #next activation neuron should be the last vector of neurons that we added
            current_activation = self.activation_layers[-1] 

        return current_activation


#random perturbation: 
#derivative of the loss v weights function would give you how much you need to change the weights to minimize
#loss
#partial derivatives in relation to each weight value: go in the value with the greatst negative slope
#for each individual weight value

#my notebook has most of the notes for the partial derivatives and formulas for its used n gradient stuff
#remember to move in the opposite direction to the gradient vector as the gradient vector points up

#weight, bias, and previous activation (which has its own weight and bias and previous activation)
#these three are all mostly independent of each other in value, related only by their product
#weights are indepdent to each other between neurons (including ones that go to the same or from the same
#activation neuron)

#chain rule is essentially just relating one unit to the other given their intermediate relations

#to take partial derivative of previous layer's activation: 
#take partial derivative in terms of cost in 
#terms of the cost to the previous activation's layer is the weight 
# between the cost and the activation layer
#need a new cost value for moving onto each previous layer, the overall trend going backwards should 
# be the same
