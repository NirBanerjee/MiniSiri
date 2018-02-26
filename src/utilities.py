from __future__ import division
import numpy as np
import math

def expCalculator(value):
	return math.exp(value)

def logCalculator(value):
	return math.log(value)

def calculateInnerProd(thetaDict, feature):
	innerProd = 0.0
	for key in feature:
		if key in thetaDict:
			innerProd = innerProd + thetaDict[key] * 1
	return innerProd

def getLikelihoodEstimate(theta, feature, classSet):
	likelihoods = []
	for label in classSet:
		likelihoods.append(expCalculator(calculateInnerProd(theta[label], feature)))
	likelihoods = np.array(likelihoods)
	return likelihoods/np.sum(likelihoods)

def getGradient(likelihoodList, actualLabel):	
	likelihoods_ = []
	i = 0
	for likelihood in likelihoodList:
		likelihood = -likelihood
		if i == actualLabel:
			likelihood +=1
		likelihoods_.append(likelihood)
		i = i+1
	return likelihoods_

def updateTheta(theta, gradients, feature,cmapinv,eta):
	i = 0
	for gradient in gradients:
		for ft in feature:
			theta[cmapinv[i]][ft] = theta[cmapinv[i]][ft] + eta*gradient
		i = i+1
	return theta

def getErrorRate(featureList, theta, classSet, classMap, labelData):
	acc = 0.0
	featureCount = 0
	for i in range(len(featureList)):
		feature = featureList[i]
		if len(feature) > 0:
			featureCount = featureCount + 1;
			likelihoodList = getLikelihoodEstimate(theta, feature, classSet)
			if np.argmax(likelihoodList) == classMap[labelData[i]]:
				acc = acc+1

	return (1-(acc/featureCount))

def getPredictedLabels(featureList, theta, classSet, classMapInv):
	predictedLabels = []
	for i in range(len(featureList)):
		feature = featureList[i]
		if len(feature) > 0:
			likelihoodList = getLikelihoodEstimate(theta, feature, classSet)
			predictedLabels.append(classMapInv[np.argmax(likelihoodList)])
		else:
			predictedLabels.append("")
	return predictedLabels
