# my first little neuron 
import math

def sigmoid(x):
    y = (1.0)/(1+math.exp(-1*x))
    return y

def activate(inputs,weights):
    # perform net input
    # zip function basically map x and w to input and weights
    h=0
    for x,w in zip(inputs,weights):
        h += x*w 
    #perform the activation
    return sigmoid(h)
    
if __name__ == "__main__":
     inputs = [.5,.3,.2]
     weights = [.4,.7,.3]
     output = activate(inputs,weights) #computations unit of neuron 
     print(output)