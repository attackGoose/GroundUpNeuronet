import numpy as np

class Encoder:
    #encoder layer of a transformer
    def __init__(self, input_size):
        pass

#transformer is really simple in retrospect, its just a ton of normalization layers, feed forward, 
#and multi-headed attention layers, so i'll make it myself here in a month and then use it to make something
#doing this to get intuition and use this to apply for research cus indepth knowledge of math is needed

class WordEmbeddings:
    def __init__(self, inputs: str, query_size): 
        """
        #tokenize the string
        #for now, this process will give each of them a random value, change value later as it trains, each word
        #has a trainable parameter that changes the word's meaning
        #the shape has to be a really high dimension otherwise a ton of words would be closely matche diwth
        #one another as there are not enough dimensions to fit all the definitions
        #two vectors of their actual value (what the word actually means), key (to refer to them, the word itself)
        #and a query of the actual meanings of the word
        #how many values to represent meaning of word in context or the context that it is used in 
        """

        #if its done in this way all the values including key, value, and query are all going to be random
        #self.embeddings = [np.random.rand(3, query_size) for input in inputs] 
        #too many parameters if upscaled, do something else
        pass


    def tokenize(self):
        pass #make each word into a token here

    def positionEmbedding(self):
        pass #add position embeddings to the tokenized words, essentially adding the value and query vectors

    