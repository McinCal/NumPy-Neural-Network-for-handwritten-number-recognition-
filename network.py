# network.py is used to house the main body of the neural network


import numpy as np

    
def reset_parameters(neurons_in, neurons_hidden, neurons_out, seed = None): #sets all the values to as if they were starting fresh
    np.random.seed(seed)
    W1, W2 = randomize_weights(neurons_in, neurons_hidden, neurons_out)
    b1 = np.zeros(neurons_hidden)
    b2 = np.zeros(neurons_out)
    return W1,W2,b1,b2

    
def randomize_weights(neurons_in, neurons_hidden, neurons_out):
    W1 = np.random.normal(size=(neurons_in, neurons_hidden))  # every value in the matrix is random from a normal distribution
    W2 = np.random.normal(size=(neurons_hidden, neurons_out)) 
    
    for i in range(len(W1)):
        for j in range(len(W1[i])):
            W1[i,j] *= np.sqrt(2 / neurons_in) # scales every value down
    for i in range(len(W2)):
        for j in range(len(W2[i])):
            W2[i,j] *= np.sqrt(2 / neurons_hidden) # scales every value down
    
    return W1, W2


def one_hot(labels): #used to return an array of size N x 10, giving the correct label of each but of data N as a list (e.g. 3 => '0001000000')
    N = len(labels)
    targets = np.zeros((N, 10))
    targets[np.arange(N), labels] = 1
    return targets


def forward_pass(input_data,W1,W2,b1,b2):
    z_hidden = input_data @ W1 + b1 # gets the current value for the hidden layer neurons using the formula z = yw + b (z = activation, y = input data, w = weight, b = bias)
    a_hidden = 1 / (1 + np.exp(-z_hidden)) # activated: sigmoid function to 'squeeze' each value of z between 0 and 1
    z_out = a_hidden @ W2 + b2 # gets the output values based on the hidden layer values using the same formula
    a_out = 1 / (1 + np.exp(-z_out)) # sigmoid again
    return z_hidden, a_hidden, z_out, a_out


def get_cost(a_out,labels):
    correct_values = one_hot(labels)
    N = len(correct_values)
    cost = ((a_out - correct_values)**2) /N # gets the difference of each probability to its expected value and squares it
    return np.sum(cost)

def back_pass(input_data, a_hidden, a_out, labels, W2):
    correct_values = one_hot(labels)
    N = len(labels)
    # here comes the big maths. see notes for working out derivatives 
    
    # chain:     (b1, w1, input_data) -> z_hidden,    z_hidden -> a_hidden,    (b2, w2, a_hidden) -> z_out,    z_out -> a_out,    a_out -> cost 


    # dcost/da_out 
    dcost_wrt_a_out = (2 / N) * (a_out - correct_values)

    # da_out/dz_out 
    da_out_wrt_z_out = a_out * (1 - a_out)

    # dcost/dz_out
    dcost_wrt_z_out = dcost_wrt_a_out * da_out_wrt_z_out

    # dcost/b2, dcost/w2, dcost/da_hidden (by subbing in above formula)
    dcost_wrt_b2 = dcost_wrt_z_out.sum(axis = 0)

    dcost_wrt_W2 = a_hidden.T @ dcost_wrt_z_out # transposed so that cross product sizes fit 

    dcost_wrt_a_hidden = dcost_wrt_z_out @ W2.T
    
    # da_hidden/dz_hidden, dcost/dz_hidden 
    da_hidden_wrt_z_hidden = a_hidden * (1-a_hidden)

    dcost_wrt_z_hidden = dcost_wrt_a_hidden * da_hidden_wrt_z_hidden

    # dcost/dw1, dcost/db1
    dcost_wrt_W1 = input_data.T @ dcost_wrt_z_hidden

    dcost_wrt_b1 = dcost_wrt_z_hidden.sum(axis = 0)

    
    return dcost_wrt_W1, dcost_wrt_W2, dcost_wrt_b1, dcost_wrt_b2


def update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate):
    W1 = W1 - dW1 * learning_rate # negative because dW1 is where the cost increases, but we want the downwards gradient to minimise cost
    W2 = W2 - dW2 * learning_rate
    b1 = b1 - db1 * learning_rate
    b2 = b2 - db2 * learning_rate
    return W1, W2, b1, b2

def get_predictions(input_data, W1, W2, b1, b2):
    z_hidden, a_hidden, z_out, a_out = forward_pass(input_data,W1,W2,b1,b2)
    predictions = np.argmax(a_out, axis=1) # gets the maximum value of every row (the highest digit activation for each drawing a.k.a the prediction) as an index 
    return predictions

def get_accuracy(predictions, labels):
    accuracy = np.mean(predictions == labels) # gets the mean of how many times the prediction matched the label
    return accuracy

def save_parameters(filename, W1, W2, b1, b2):
    np.savez(filename, W1=W1, W2=W2, b1=b1, b2=b2)

def load_parameters(filename):
    with np.load(filename) as saved:
        return saved["W1"], saved["W2"], saved["b1"], saved["b2"]








    
    

