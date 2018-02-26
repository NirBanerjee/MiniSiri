import math
#Generate the sparse matrix for model1
def sparseModel1(dataFrame):
	labelList = []
	featureList = []
	for row in dataFrame:
		featureRow = {}
		if len(row) == 2:
			labelList.append(row[1])
			featureRow["bias"] = 1
			featureRow["curr:" + row[0]] = 1
			featureList.append(featureRow)
		else:
			labelList.append(None)
			featureList.append(None)

	return labelList,featureList

#Generate the sparse matrix for model2
def sparseModel2(dataFrame):
	labelList = []
	featureList = []

	featureRow = {}
	row = dataFrame[0];
	featureRow["bias"] = 1
	featureRow["curr:"+row[0]] = 1
	featureRow["prev:BOS"] = 1
	nextRow = dataFrame[1];
	featureRow["next:"+ nextRow[0]] = 1
	labelList.append(row[1])
	featureList.append(featureRow);

	length = len(dataFrame)
	for i in range(1, length - 1):
		preRow = dataFrame[i-1];
		nextRow = dataFrame[i+1];
		curRow = dataFrame[i];
		featureRow = {}
		if len(curRow) == 2:
			if len(preRow) == 2:
				prevLabel = "prev:" + preRow[0]
			elif len(preRow) == 0:
				prevLabel = "curr:BOS"
			if len(nextRow) == 0:
				nextLabel = "next:EOS"
			elif len(nextRow) == 2:
				nextLabel = "next:" + nextRow[0];

			featureRow["bias"] = 1
			featureRow["curr:" +curRow[0]] = 1
			featureRow[prevLabel] = 1
			featureRow[nextLabel] = 1

			featureList.append(featureRow);
			labelList.append(curRow[1])
		else:
			labelList.append(None)
			featureList.append(None)

	if len(dataFrame[length-1]) != 0:
		featureRow = {}
		featureRow["bias"] = 1
		row = dataFrame[length-1];
		featureRow["curr:"+row[0]] = 1
		preRow = dataFrame[length-2];
		featureRow["prev:" + preRow[0]] = 1
		nextRow = dataFrame[1];
		featureRow["next:EOS"] = 1
		labelList.append(row[1])
		featureList.append(featureRow);

	return labelList,featureList