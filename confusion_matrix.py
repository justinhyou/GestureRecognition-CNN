import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import csv
import os

#path to predictions.csv
predictions = os.path.abspath("/Users/clinic1718/Desktop/GestureRecognition-CNN/results/JIGSAW/predictions.csv")
#path to test folder
validations = os.path.abspath("/Users/clinic1718/Desktop/FramesExecution/data/validationastest/test")

#array to accumulate labels
talliedOrder = np.array([])
with open(predictions, 'r') as predictionsFile:
	predictMatrix = predictionsFile.readlines()[1:]
	index = 0
	for row in predictMatrix:
		row = row.strip()
		talliedOrder = np.append(talliedOrder, int(row[-1]))
		index+=1

#true labels
#correctedLabels = np.array([])
#convert to actual labels 
"""
0 -> G1
1 -> G2
2 -> G3
3 -> G4
4 -> G5
5 -> G6
6 -> G8
7 -> G9
8 -> G10
9 -> G11
"""
# for i in talliedOrder:
# 	if (i<7):
# 		correctedLabels = np.append(correctedLabels, i+1)
# 	else:
# 		correctedLabels = np.append(correctedLables, i+2)

cm = np.empty([0, 10])

counter = 0
for folder in os.listdir(validations):
	if folder[0] != "G":
		continue
 	numImages = len(os.listdir(validations+"/"+folder))
 	temp = np.zeros(10)
 	for i in range(counter, counter+numImages):
 		label = talliedOrder[i]
 		temp[label] += 1

 	cm = np.vstack((cm, temp))
 	counter += numImages

#normalize confusion matrix
row_sums = cm.sum(axis=1)
cm = cm/row_sums[:, np.newaxis]


classes = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11])


plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
fmt = '.2f' if True else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.show()