from __future__ import division
from SparseGenerator import sparseModel1, sparseModel2
from utilities import calculateInnerProd, getLikelihoodEstimate, getGradient, updateTheta, getErrorRate, logCalculator, getPredictedLabels, getLogLikelihood
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
	trainOutputFile = sys.argv[4]
	testOutputFile = sys.argv[5]
	metricsFile = sys.argv[6]
	numEpoch = sys.argv[7]
	featureFlag = sys.argv[8]

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
	classSet = list(classSet)
	classSet.sort()

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

			if feature is None:

				continue

			for key in feature:
				theta[label][key] = 0.0
	
	outputWrite = []
	#Train theta
	for j in range(int(numEpoch)):
		featureCount = 0
		for i in range(len(trainFeatures)):
			feature = trainFeatures[i]
			if feature is not None :
				featureCount = featureCount + 1
				likelihoodList = getLikelihoodEstimate(theta, feature, classSet)
				gradient = getGradient(likelihoodList, classMap[trainLabels[i]])
				theta = updateTheta(theta, gradient, feature, classMapInv, 0.5)

		featureCount = 0
		logLikelihood = getLogLikelihood(theta, trainFeatures, trainLabels, classSet, classMap)
		outputWrite.append("epoch=" + str(j+1) + " likelihood(train):" + str(logLikelihood))

		logLikelihood = getLogLikelihood(theta, validationFeatures, validationLabels, classSet, classMap)
		outputWrite.append("epoch=" + str(j+1) + " likelihood(train):" + str(logLikelihood))
	
	#Predict Train Labels
	labelsPredicted = getPredictedLabels(trainFeatures, theta, classSet, classMapInv)
	writer = open(trainOutputFile, 'w')
	for label in labelsPredicted:
		writer.write(label)
		writer.write("\n")
	writer.close

	#Predict Test Labels
	labelsPredicted = getPredictedLabels(testFeatures, theta, classSet, classMapInv)
	writer = open(testOutputFile, 'w')
	for label in labelsPredicted:
		writer.write(label)
		writer.write("\n")
	writer.close

	#Calculate train and test error
	outputWrite.append("error(train): " + str(getErrorRate(trainFeatures, theta, classSet, classMap, trainLabels)))
	outputWrite.append("error(test): " + str(getErrorRate(testFeatures, theta, classSet, classMap, testLabels)))
	
	writer = open(metricsFile, 'w')
	for line in outputWrite:
		writer.write(line)
		writer.write("\n")
	writer.close
