import numpy as np
import matplotlib.pyplot as plt
#import functions from other programs
from network import reset_parameters, forward_pass, get_cost, back_pass, update_parameters, get_predictions, get_accuracy, save_parameters
from dataset import load_file, split_data, centre_drawing

#--------------------------------CONSTANTS--------------------------------
NEURONS_IN = 784
NEURONS_HIDDEN = 72 # this can be varied for different results, but at 2000 drawings about 72 seems to give the best results
NEURONS_OUT = 10
LEARNING_RATE = 1
EPOCHS = 500 # epochs is number of times the network goes through the whole dataset
BATCH_SIZE = 32 # how many drawings at a time get passed into the network and back propagated on
SEED = 99245245 # seed to recreate results if necessary
CENTRE = True # decides if you want centering


#--------------------------------IMPORTING OTHER VARIABLES--------------------------------
drawings, labels = load_file("dataset.npz")
if CENTRE:
    drawings = np.array([centre_drawing(i) for i in drawings])

trainDrawings, testDrawings, trainLabels, testLabels = split_data(drawings, labels, 0.1)#0.1 is the fraction of train material to test material (10% test)

W1, W2, b1, b2 = reset_parameters(NEURONS_IN, NEURONS_HIDDEN, NEURONS_OUT, seed=SEED)

trainCosts, testCosts = [], []
trainAccuracies, testAccuracies = [], []



#--------------------------------CORE LOOP--------------------------------
print("Epoch: \t\tTrain cost: \t\tTest cost: \t\tTrain accuracy: \tTest accuracy:") # to tabulate results

for i in range(EPOCHS):
    order = np.random.permutation(len(trainDrawings)) # we dont want the epochs to use the same batches in the same order
    for start in range(0, len(trainDrawings), BATCH_SIZE): # loops from 0 to the number of training drawings in steps of 32 (start = 0,32,64...)
        batchRows = order[start : start + BATCH_SIZE] # gets the order values for each iteration of the loop. if it doesnt divide fully the last batch will be smaller (remainder)
        
        batchDrawings = trainDrawings[batchRows] # gets the drawings and labels for this batch
        batchLabels = trainLabels[batchRows]

        
        #now the training loop below:
        z_hidden, a_hidden, z_out, a_out = forward_pass(batchDrawings, W1, W2, b1, b2)
        dW1, dW2, db1, db2 = back_pass(batchDrawings, a_hidden, a_out, batchLabels, W2)
        W1, W2, b1, b2 = update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, LEARNING_RATE)

        
    #show results for each epoch:
        
    # training results
    z_hidden, a_hidden, z_out, a_out = forward_pass(trainDrawings, W1, W2, b1, b2)
    
    trainCost = get_cost(a_out, trainLabels)
    trainPredictions = get_predictions(trainDrawings, W1, W2, b1, b2)
    trainAccuracy = get_accuracy(trainPredictions, trainLabels)
    
    # test results
    z_hidden, a_hidden, z_out, a_out = forward_pass(testDrawings, W1, W2, b1, b2)

    testCost = get_cost(a_out, testLabels)
    testPredictions = get_predictions(testDrawings, W1, W2, b1, b2)
    testAccuracy = get_accuracy(testPredictions, testLabels)

    #save for plotting
    trainCosts.append(trainCost)
    trainAccuracies.append(trainAccuracy)
    testCosts.append(testCost)
    testAccuracies.append(testAccuracy)
    
    if i%10 == 0: # every 10 epochs
        print("{0}\t\t{1:0.5f}\t\t\t{2:0.5f}\t\t\t{3:0.5f}\t\t\t{4:0.5f}".format(i,trainCost,testCost,trainAccuracy,testAccuracy)) # print tabulated results

        
save_parameters("digits_weights.npz", W1, W2, b1, b2) # save parameters in case of future use

#--------------------------------PLOT RESULTS--------------------------------


plt.figure()


plt.subplot(121) # making subplot with 1 row, 2 cols
plt.title('cost against epoch')
plt.plot(trainCosts,color="blue",label='training cost')
plt.plot(testCosts,color="red",label='test cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()


plt.subplot(122)
plt.title('accuracy against epoch')
plt.plot(trainAccuracies,color="blue",label='training accuracy')
plt.plot(testAccuracies,color="red",label='test accuracy,')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()



plt.tight_layout(pad=2.0) # adds a gap between sub plots

plt.savefig("training_curves.png")
plt.show()













