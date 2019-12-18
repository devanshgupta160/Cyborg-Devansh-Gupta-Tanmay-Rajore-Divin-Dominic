#Made By: Devansh Gupta, Tanmay Rajore, Divin Dominic

import numpy as np
from sklearn import datasets
import copy

#Using sklearn to import the iris dataset
iris = datasets.load_iris()

#Making it compatible with the perceptron classification so that the output is in 0 or 1 according to the perceptron.     
s=copy.deepcopy(iris.target)
for i in range(len(s)):
    if(s[i]!=0):
        s[i]=1

class Perceptron():
    """A class used to make the perceptron"""
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.1):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0           
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

#Initializing the training data(25 elements from each class)

training_inputs1=np.array(iris.data[:25])
training_inputs2=np.array(iris.data[50:75])
training_inputs3=np.array(iris.data[100:125])
labels1_1=np.array(s[:25]).transpose()
labels1_2=np.array(s[50:75]).transpose()
labels1_3=np.array(s[100:125]).transpose()
#Initializing the perceptron

p=Perceptron(no_of_inputs=4)

#Train
for i in range(5):
    p.train(training_inputs2,labels1_2)
    p.train(training_inputs1,labels1_1)
    p.train(training_inputs3,labels1_3)

#Calculating accuracy using the rest of the data
count=0
count1=0
count2=0
for i in range(25,50):
    if(s[i]==p.predict(np.array(iris.data[i]))):
        count1+=1
for i in range(75,100):
    if(s[i]==p.predict(np.array(iris.data[i]))):
        count1+=1
for i in range(125,150):
    if(s[i]==p.predict(np.array(iris.data[i]))):
        count1+=1

#Calculating the accuracy
print("Accuracy for p: " + str(count1*4/3))

#Providing a way to manually test the data in order to check
for i in range(25,50):
    print(str(iris.data[i]) + "||" + str(s[i]) + "||" + str(iris.target[i]))
for i in range(75,100):
    print(str(iris.data[i]) + "||" + str(s[i]) + "||" + str(iris.target[i]))
for i in range(125,150):
    print(str(iris.data[i]) + "||" + str(s[i]) + "||" + str(iris.target[i]))

a=float(input("Enter first parameter: "))
b=float(input("Enter second parameter: "))
c=float(input("Enter third parameter: "))
d=float(input("Enter fourth parameter: "))

print(p.predict(np.array([a,b,c,d])))

