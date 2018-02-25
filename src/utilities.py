from __future__ import division
import numpy as np
import math

def expCalculator(value):
	return math.exp(value)

def calculateInnerProd(thetaDict, feature):
	innerProd = 0.0
	for key in feature:
		innerProd = innerProd + thetaDict[key] * 1

	return innerProd

def getLikelihoodEstimate(theta, feature, classSet):
	normalizer = 0.0
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
		i+=1

	return likelihoods_

def updateTheta(theta, gradients, feature,cmapinv,eta):
	i = 0
	for gradient in gradients:
		for ft in feature:
			theta[cmapinv[i]][ft] = theta[cmapinv[i]][ft] + eta*gradient
		i+=1
	return theta
