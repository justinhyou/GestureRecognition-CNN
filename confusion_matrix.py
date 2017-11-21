import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import csv
import os

#path to predictions.csv
predictions = os.path.abspath("/Users/clinic1718/Desktop/FramesExecution/results/JIGSAW/Normalized_validation/predictions.csv")
#path to test folder
validations = os.path.abspath("/Users/clinic1718/Desktop/normFrames80/test/test_validation")

#array to accumulate labels
# talliedOrder = np.array([])
# with open(predictions, 'r') as predictionsFile:
# 	predictMatrix = predictionsFile.readlines()[1:]
# 	index = 0
# 	for row in predictMatrix:
# 		row = row.strip()
# 		talliedOrder = np.append(talliedOrder, int(row[-1]))
# 		index+=1

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

pathToGestures = "/Users/clinic1718/Desktop/normFrames80/test/test_validation"

gestureDict = {}

folderDict = {
	"G1":0,"G2":1,"G3":2,"G4":3,"G5":4,"G6":5,"G8":6,"G9":7,"G10":8,"G11":9
}
#assumes the only things in pathToGestures are folders with names corresponding to gestures,
#which each contain only image files

for folder in os.listdir(pathToGestures):
	#print folder
	if os.path.isdir(pathToGestures+ "/" + folder):
		for file in os.listdir(pathToGestures + "/" + folder):
			#print file, folder
			gestureDict[file[:-4]] = folderDict[folder]
			#print file[:-4], gestureDict[file[:-4]]

#cm = [[0 for item in range(10)] for item in range(10)]
cm = np.zeros((10, 10))
with open(predictions) as csv:
	csv = [line.split(",") for line in csv][1:]
	# for row in csv:
	# 	predict = int(row[-1])
	# 	print gestureDict[row[0]], predict

	#print gestureDict

	for row in csv:
		#print row[0]
		true = gestureDict[row[0]]
		predict = int(row[-1])
		#cm[predict][true] += 1
		cm[true][predict] += 1

# cm = np.empty([0, 10])

# counter = 0
# for folder in os.listdir(validations):
# 	if folder[0] != "G":
# 		continue
#  	numImages = len(os.listdir(validations+"/"+folder))
#  	temp = np.zeros(10)
#  	for i in range(counter, counter+numImages):
#  		label = talliedOrder[i]
#  		temp[label] += 1

#  	cm = np.vstack((cm, temp))
#  	counter += numImages

#normalize confusion matrix
row_sums = cm.sum(axis=1)
cm = cm/row_sums[:, np.newaxis]


classes = np.array(["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"])


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