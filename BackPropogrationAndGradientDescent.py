import numpy as np

# save the activations and the derivatives 
# implement back propogation 
# Implement gradient descent
# implement a train method 
# train our net with some dummy data sets
# make some predictions 

class MLP(object):

    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        
        activations = []
        for i in range(len(layers)):
            #one demensional array
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations= activations 
        
        derivatives = []
        # three layers has only two wt matrices 
        for i in range(len(layers)-1):
            #one demensional array
            # no of neuron in layer i and no of neurons in layer i+1
            d = np.zeros(layers[i],layers[i+1])
            activations.append(d)
        self.derivatives= derivatives 
        
        
            

   # computation function which gives prediction
    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs
        #activations for first layer 
        self.activations[0] = inputs

        # iterate through the network layers
        for i,w in enumerate(self.weights):

            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
            

        # return output layer activation
        return activations


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        
        y = 1.0 / (1 + np.exp(-x))
        return y
    
    def back_propogate(self,error):
        
        # s is nothing but the sigmoid function 
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error*self._sigmoid_derivatives(activations)
            current_activations = self.activations[i]
            self.derivatives[i] = np.dot(current_activations,delta)
            
            
            
    def _sigmoid_derivatives(self,x):
        return x * (1.0 - x) 
            


if __name__ == "__main__":

    # create a Multilayer Perceptron
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagate(inputs)

    print("Network activation: {}".format(output))
    # next we will train out network