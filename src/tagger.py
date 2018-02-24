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
			dataFrame.append(tuple(row))
	return dataFrame

#Main Method
if __name__ == '__main__':
	
	trainFileName = sys.argv[1];
	dataFrame = openDataFile(trainFileName)
	#print dataFrame
	#sparseData = {}
	sparseData = sparseModel1(dataFrame)
	print sparseData