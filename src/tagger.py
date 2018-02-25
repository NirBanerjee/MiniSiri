from __future__ import division
from SparseGenerator import sparseModel1, sparseModel2
from utilities import calculateInnerProd, getLikelihoodEstimate, getGradient, updateTheta, getErrorRate
import math
import csv
import sys
import numpy as np

#Method to open file to read data
def openDataFile(fileName):
	dataFrame = []
	with open(fileName, 'rb') as csvfile:
		fileData = csv.reader(csvfile, delimiter='\t')
		for row in fileData:
			dataFrame.append(row)
	return dataFrame

#Main Method
if __name__ == '__main__':
	
	trainFileName = sys.argv[1];
	validationFileName = sys.argv[2]
	testFileName = sys.argv[3]
	numEpoch = sys.argv[4]
	featureFlag = sys.argv[5]

	trainData = openDataFile(trainFileName)
	validationData = openDataFile(validationFileName)
	testData = openDataFile(testFileName)

	if int(featureFlag) == 1:
		trainLabels, trainFeatures = sparseModel1(trainData)
		validationLabels, validationFeatures = sparseModel1(validationData)
		testLabels, testFeatures = sparseModel1(testData)
	else:
		trainLabels, trainFeatures = sparseModel2(trainData)
		validationLabels, validationFeatures = sparseModel2(validationData)
		testLabels, testFeatures = sparseModel2(testData)

	#Get Count of Classes
	classSet = set(trainLabels)

	#Create Mapping of labels and indices
	classMap = {}
	i = 0
	for cl in classSet:
		classMap[cl] = i 
		i = i+1

	#Create Inverse Mapping of labels and indices
	classMapInv = {}
	i = 0
	for cl in classSet:
		classMapInv[i] = cl 
		i = i+1

	#Initialize theta matrix
	theta = {}

	for label in classSet:
		theta[label] = {}
		for feature in trainFeatures :
			for key in feature:
				theta[label][key] = 0.0

	#Train theta
	for j in range(int(numEpoch)):
		for i in range(len(trainFeatures)):
			feature = trainFeatures[i]
			likelihoodList = getLikelihoodEstimate(theta, feature, classSet)
			gradient = getGradient( likelihoodList, classMap[trainLabels[i]])
			theta = updateTheta(theta, gradient, feature, classMapInv, 0.5)
		print "Epoch=" + str(j+1) + "Error (train): " + str(getErrorRate(trainFeatures, theta, classSet, classMap, trainLabels))
		print "Epoch=" + str(j+1) + "Error (test): " + str(getErrorRate(testFeatures, theta, classSet, classMap, testLabels))

	
	
	




	






