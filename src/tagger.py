from __future__ import division
from SparseGenerator import sparseModel1, sparseModel2
from utilities import calculateInnerProd, getLikelihoodEstimate, getGradient, updateTheta
import math
import csv
import sys

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
	dataFrame = openDataFile(trainFileName)

	labelData,featureList = sparseModel1(dataFrame)
	#labelData,featureList = sparseModel2(dataFrame)

	#Get Count of Classes
	classSet = set(labelData)

	classMap = {}
	i = 0
	for cl in classSet:
		classMap[cl] = i 
		i = i+1

	classMapInv = {}
	i = 0
	for cl in classSet:
		classMapInv[i] = cl 
		i = i+1

	#Initialize theta matrix
	theta = {}

	for label in classSet:
		theta[label] = {}
		for feature in featureList:
			for key in feature:
				theta[label][key] = 0.0

	for i in range(len(featureList)):
		feature = featureList[i]
		likelihoodList = getLikelihoodEstimate(theta, feature, classSet)
		gradient = getGradient( likelihoodList, classMap[labelData[i]])
		theta = updateTheta(theta, gradient, feature, classMapInv, 0.5)
		print theta


	






