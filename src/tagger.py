from __future__ import division
from SparseGenerator import sparseModel1, sparseModel2
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
	for val in labelData:
		print val


	print "+++++++++++++++++++++++++"
	labelData,featureList = sparseModel2(dataFrame)
	for val in labelData:
		print val