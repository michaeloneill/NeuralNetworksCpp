import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import ceil, sqrt

case = 1

# Note that the 1st learned basis for cases 1-3 assumes 25 units in the first hidden layer, and that for case 4 assumes 20 5*5 filters.

# plot the first 100 images 

file = open("outputDigits", "r")

images = np.loadtxt(file).reshape(100, 28, 28)

fig = plt.figure("First 100 digits")
for x in range(10):
    for y in range(10):
        ax = fig.add_subplot(10, 10, 10*x+y+1) # note grid filled col wise
        ax.matshow(images[10*x+y], cmap = cm.binary) 
        plt.xticks([])
        plt.yticks([])
plt.show()

file.close()


# plot the cost histories

file = open("outputNNCostHistory", "r")

costHistory = np.loadtxt(file)

fig = plt.figure("Cost History")

plt.plot(costHistory[:, 0], costHistory[:, 1])
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

file.close()

if ((case == 1) or (case == 2)):

    # plot the first learned basis for case 1 or 2
    
    file = open("outputNNBasisLayer1", "r")
    
    basis1 = np.loadtxt(file).reshape(25, 28, 28)
    
    fig = plt.figure("Visualisation of First Learned Basis")
    
    for x in range(5):
        for y in range(5):
            ax = fig.add_subplot(5, 5, 5*x+y+1) # note grid filled col wise
            ax.matshow(basis1[5*x+y], cmap = cm.binary) 
            plt.xticks([])
            plt.yticks([])
    plt.show()
            
    file.close()

            
    # plot the second learned basis for case 1 or 2

    file = open("outputNNBasisLayer2", "r")
    
    basis2 = np.loadtxt(file).reshape(10, 5, 5)

    fig = plt.figure("Visualisation of Second Learned Basis")
    
    for x in range(2):
        for y in range(5):
            ax = fig.add_subplot(5, 2, 5*x+y+1) # note grid filled col wise
            ax.matshow(basis2[5*x+y], cmap = cm.binary) 
            plt.xticks([])
            plt.yticks([])
    plt.show()
            
    file.close()


elif (case == 3):

    # just plot 1st learned basis for case 3
    
    file = open("outputNNBasisLayer1", "r")
    
    basis1 = np.loadtxt(file).reshape(25, 28, 28)
    
    fig = plt.figure("Visualisation of First Learned Basis")
    
    for x in range(5):
        for y in range(5):
            ax = fig.add_subplot(5, 5, 5*x+y+1) # note grid filled col wise
            ax.matshow(basis1[5*x+y], cmap = cm.binary) 
            plt.xticks([])
            plt.yticks([])
    plt.show()
            
    file.close()


    
elif (case == 4):

            
    # Just plot fist learned basis for case 4
    
    file = open("outputNNBasisLayer1", "r")
    
    basis1 = np.loadtxt(file).reshape(20, 5, 5)
    fig = plt.figure("Visualisation of First Learned Basis")
    
    for x in range(5):
        for y in range(4):
            ax = fig.add_subplot(4, 5, 4*x+y+1) # note grid filled col wise
            ax.matshow(basis1[4*x+y], cmap = cm.binary) 
            plt.xticks([])
            plt.yticks([])
    plt.show()
            
    file.close()

else:
    pass # dont plot basis for any other case
            



# plot the learning curves

file = open("outputLC", "r")

data = np.loadtxt(file)
m = len(data)

plt.figure('learning curves')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('proportion training samples used')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()

# plot validation curves

file = open("outputLamVal", "r")

data = np.loadtxt(file)

plt.figure('Lambda validation')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('lambda')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()

file = open("outputAlphaVal", "r")

data = np.loadtxt(file)

plt.figure('Alpha validation')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('alpha')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()



