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

pathToGestures = "/Users/clinic1718/Desktop/normFrames80/test/test_validation"

gestureDict = {}

# folderDict = {
# 	"G1":0,"G2":1,"G3":2,"G4":3,"G5":4,"G6":5,"G8":6,"G9":7,"G10":8,"G11":9
# }

folderDict = {
	"G1":0,"G2":3,"G3":4,"G4":5,"G5":6,"G6":7,"G8":8,"G9":9,"G10":1,"G11":2
}

#assumes the only things in pathToGestures are folders with names corresponding to gestures,
#which each contain only image files
for folder in os.listdir(pathToGestures):
	#print folder
	if os.path.isdir(pathToGestures+ "/" + folder):
		for file in os.listdir(pathToGestures + "/" + folder):
			gestureDict[file[:-4]] = folderDict[folder]

cm = np.zeros((10, 10))
with open(predictions) as csv:
	csv = [line.split(",") for line in csv][1:]

	for row in csv:
		trueLabel = gestureDict[row[0]]
		predictedLabel = int(row[-1])
		cm[predictedLabel][trueLabel] += 1



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