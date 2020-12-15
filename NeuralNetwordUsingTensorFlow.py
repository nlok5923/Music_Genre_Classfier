"""import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Tensor flow is amazing DL library we will be using to build neueal network

# datasets for arithmatic sum
# inputs 
# array([[0.1,0.2],[0.1,0.2]])
# outputs
# array([[0.3],[0.3]])
# Python automatically stores the value of the last expression in the interpreter to a particular variable called "_." You can also assign these value to another variable if you want.

def generate_dataset(num_samples, test_size):
    
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples) ])
    y = np.array([[i[0]+i[1]] for i in x])


# print(x)
# print(y)
# spliting data set into training set and test set 
 
# training set will be of all 30% of inputs 
    x_train , x_test , y_train , y_test =  train_test_split(x,y,test_size=test_size)
    return x_train,x_test,y_train, y_test 
# steps 
# build model 
# compile model 
# train model
# evaluate model
# make predictions 
if __name__ == "__main__":
    x_train , x_test , y_train , y_test = generate_dataset(5000,0.3)
   # print("x_test \n {}",format(x_test))
   # print("x_train \n {}",format(x_train))
   # print("y_test \n {}",format(y_test))
   # print("y_train \n {}",format(y_train))
   
   #input signal movement will be from left to right as we used Sequential 
   # build model 2->5->1
   # 2 (input layer) 5(hidden layer) 1 output layer
   model = tf.keras.Sequential([
           # first hidden layer with 5 neuron , no of neuronin input layer
           tf.keras.Dense(5,input_din = 2, activation = "sigmoid"),
           tf.keras.Dense(1, activation="sigmoid")]
           )
   # compiling model
   # MSE: min squre error
   # SGD = stochastic gradient descent
   optimizer = tf.keras.optimizer.SGD(learning_rate = 0.1)
   model.compile(optimizer =optimizer,loss="MSE")
   
   # train model
   model.fit(x_train, y_train, epochs = 100)
   
   # evaluate model
   print("Model evaluation: " )
   #writing vebose to get report
   model.evaluate(x_test, y_test,verbose = 1)
   
   # making predictions 
   data =np.array([[0.1,0.2],[0.2,0.2]])
   predictions = model.predict(data)
   print("Some predictions: ")
   
   # here hamne model ko arithmatic addtion karna sikha diya
   for d, p in zip(data, predictios):
       print("{}+ {}= {}", formtat(d[0],d[1],p[0]))"""
       
     
# above is mine shit 
       # below is his code 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from random import random


def generate_dataset(num_samples, test_size=0.33):
    """Generates train/test data for sum operation
    :param num_samples (int): Num of total samples in dataset
    :param test_size (int): Ratio of num_samples used as test set
    :return x_train (ndarray): 2d array with input data for training
    :return x_test (ndarray): 2d array with input data for testing
    :return y_train (ndarray): 2d array with target data for training
    :return y_test (ndarray): 2d array with target data for testing
    """

    # build inputs/targets for sum operation: y[0][0] = x[0][0] + x[0][1]
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    # split dataset into test and training sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # create a dataset with 2000 samples
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)

    # build model with 3 layers: 2 -> 5 -> 1
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
      tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # choose optimiser
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    # compile model
    model.compile(optimizer=optimizer, loss='mse')

    # train model
    model.fit(x_train, y_train, epochs=100)

    # evaluate model on test set
    print("\nEvaluation on the test set:")
    model.evaluate(x_test,  y_test, verbose=2)

    # get predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    # print predictions
    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))
   
   
   
   
   
   
    
    