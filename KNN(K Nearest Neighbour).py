##  SUPERVISED LEARNING #

#KNN - K Nearest Neighbour Algorithm

import csv
with open(r"D:\CSV Files\Iris.csv") as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print(','.join(row))

# load dataset into test and training datasets
import csv
import random
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(r"D:\CSV Files\Iris.csv") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = (dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

trainingSet=[]
testSet=[]
loadDataset(r'D:\CSV Files\Iris.csv',0.66, trainingSet, testSet)
print('Train:' + repr(len(trainingSet)))
print('Test:' + repr(len(testSet)))

#similarity
#here we can directlu use the Euclidean distance

import math
def euclideanDistance(instance1, instance2, length):
    """instance1 and 2 are the 2 points for which we want to
    find the euclidean distance"""
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance:' + repr(distance), "This is the Euclidean Distance")

##Now we look for K nearest Neighbours
#we use function as getneighbours

import operator
def getNeighbours(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

trainSet = [[2,2,2, 'a'],[4,4,4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbours = getNeighbours(trainSet, testInstance, 1)
print(neighbours)

#now we def get response
import operator
def getResposne(neighbours):
    ClassVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in ClassVotes:
            ClassVotes[response] += 1
        else:
            ClassVotes[response] =1
        sortedVotes = sorted(ClassVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

neighbours = [[1,1,1, 'a'],[2,2,2,'a'],[3,3,3,'b']]
print(getResposne(neighbours))

#Now we Evaluate the accuracy
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
        return (correct/float(len(testSet))) * 100.0

testSet = [[1,1,1, 'a'],[2,2,2,'a'],[3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)

# combine into main function
def main():
    #prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset('D:\CSV Files\Iris.csv',split, trainingSet, testSet)
    print('Train Set:' + repr(len(trainingSet)))
    print('Test Set:' + repr(len(testSet)))
    #generate Predictions
    predictions = []
    k=3
    for x in range(len(testSet)):
        neighbours = getNeighbours(trainingSet, testSet[x], k)
        result = getNeighbours(neighbours)
        predictions.append(result)
        print('> predicted =' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:' + repr(accuracy) + '%')
main()
