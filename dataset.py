
import numpy as np


def load_file(filename):
    data = np.load(filename)
    drawings = data["drawings"]
    labels = data["labels"]
    data.close()
    return drawings, labels

def split_data(drawings, labels, test_fraction): # split the data into seperate arrays of training vs testing data
    trainDrawings = []
    testDrawings = []
    trainLabels = []
    testLabels = []
    np.random.seed(1) # seed so we can recreate results if necessary
    
    for num in range(10):
        iList = []
        
        for i in range(len(labels)):
            if labels[i] == num:
                iList.append(i)
                
        np.random.shuffle(iList) # shuffle the order of the data so that its different drawings used for testing and training each time
        count = len(iList)
        testNum = round(count*test_fraction)
        
        for j in range(count):
            if j<testNum:
                testDrawings.append(drawings[iList[j]])
                testLabels.append(int(labels[iList[j]]))
            else:
                trainDrawings.append(drawings[iList[j]])
                trainLabels.append(int(labels[iList[j]]))
                
    # below lines shuffles both parallel arrays in the same order
    order = np.random.permutation(len(trainDrawings))
    # coverts to numpy arrays which are basically matrices not lists
    trainDrawings = np.array(trainDrawings)
    testDrawings = np.array(testDrawings)
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)
    
    trainDrawings = trainDrawings[order] # re orders to the shuffled order
    trainLabels = trainLabels[order]
    
    return trainDrawings, testDrawings, trainLabels, testLabels
        
def centre_drawing(drawing):
    # back to 2D array
    grid = drawing.reshape(28, 28)
    total = grid.sum()

    # get 'centre of mass'
    rowCentre = 0
    colCentre = 0
    for row in range(28):
        for col in range(28):
            rowCentre += grid[row][col]*row
            colCentre += grid[row][col]*col
    rowCentre /= total
    colCentre /= total

    # get shift. middle = 13.5 (halfway between cells 13 and 14)
    rowShift = int(round(rowCentre - 13.5))
    colShift = int(round(colCentre - 13.5))
    

    shiftedGrid = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            if i+rowShift < 0 or i+rowShift > 27 or j+colShift < 0 or j+colShift > 27: # if the pixel is getting shifted out of the grid, set it to 0 (no rolling)
                shiftedGrid[i][j] = 0
            else:
                shiftedGrid[i][j] = grid[i+rowShift][j+colShift]
                
    shiftedGrid.flatten()
    
    return shiftedGrid.flatten()
            








            
    
