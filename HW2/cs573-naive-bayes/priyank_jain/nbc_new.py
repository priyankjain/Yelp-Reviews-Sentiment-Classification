###############################################################################
#Author: Priyank Jain (@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import shuffle
import string
import codecs
import operator
from collections import OrderedDict
import random
class selector(object):
	def __init__(self, file):
		if not os.path.exists(file):
			raise FileNotFoundError(file + ' does not exist')
		self.file = file
		
	def read(self, percentage):
		self.percentage = percentage
		data = open(self.file, 'r').read()
		data = data.replace('\n ', ' ')
		databits = data.split('\n')
		databits = databits[:-1]
		self.data = []
		for bit in databits:
			cols = bit.strip().split('\t')
			label = int(cols[1])
			text = cols[2].strip()
			self.data.append({'text': text, 'label': label})
		shuffle(self.data)
		divider = int(percentage*len(self.data)/100)
		return self.data[:divider], self.data[divider:]	

class preprocessor(object):
	def __init__(self):
		pass

	def process(self, data):
		self.data = data
		for dct in self.data:
			dct['text'] = dct['text'].lower()
			dct['text'] = dct['text'].translate(str.maketrans('','',string.punctuation))
			dct['text'] = dct['text'].split()
		return self.data

def comparator(x, y):
	if x[1] < y[1]:
		return -1
	elif x[1] > y[1]:
		return 1
	else:
		return random.choice([-1, 1])

def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0  
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

class transformer(object):
	def __init__(self, data, featureCount, stopwordsCount=100):
		self.data = data
		self.featureCount = featureCount
		self.stopwordsCount = stopwordsCount
		self.feature_ordered_set = None

	def histogram(self):
		hist = dict()
		for dct in self.data:
			reviewHist = dict()
			for words in dct['text']:
				reviewHist[words] = 1
			for word in reviewHist.keys(): 
				hist.setdefault(word, 0)
				hist[word] += 1
		return hist

	def transform(self):
		hist = self.histogram()
		sorted_hist = sorted(hist.items(), key = \
			cmp_to_key(comparator), reverse=True)
		sorted_hist = sorted_hist[self.stopwordsCount:]		
		for i in range(0,10):
			print("WORD{0}".format(i+1), sorted_hist[i][0])
		sorted_hist = dict(sorted_hist[:self.featureCount])
		feature_set = set(sorted_hist.keys())
		feature_ordered_set = OrderedDict([(feature, 0) for feature in feature_set])
		for dct in self.data:
			featureDict = OrderedDict(\
				[(feature, 0) for feature in feature_ordered_set.keys()])
			for word in dct['text']:
				if word in feature_set:
					featureDict[word] = 1
			dct['features'] = list(featureDict.values())
			dct.pop('text', None)
		self.feature_ordered_set = feature_ordered_set
		return self.data

	def transform_test(self, test_data):
		for dct in test_data:
			featureDict = OrderedDict(\
				[(feature, 0) for feature in self.feature_ordered_set.keys()])
			for word in dct['text']:
				if word in self.feature_ordered_set.keys():
					featureDict[word] = 1
			dct['features'] = list(featureDict.values())
			dct.pop('text', None)
		return test_data

class NaiveBayesModel(object):
	def __init__(self, smoothing=True, numLabels=2):
		self.smoothing = smoothing
		self.numLabels = numLabels
		self.priors = dict()
		self.conditionals = dict()

	def train(self, data):
		for dct in data:
			self.priors.setdefault(dct['label'], 0)
			self.priors[dct['label']] += 1
			featuresCount = len(dct['features'])
			self.conditionals.setdefault(dct['label'], [0 for x in range(0, featuresCount)])
			self.conditionals[dct['label']]  = [x+y for x,y in \
				zip(self.conditionals[dct['label']], dct['features'])]

	def priorProb(self, y):
		return self.priors[y]/sum(self.priors)

	def conditionalProb(self, x, y):
		return (self.conditionals[y][x]+1)/(self.priors[y]+2)

	def defaultPrediction(self):
		max_label = None
		max_prior = 0
		for label, prior in self.priors.items():
			if max_prior < prior:
				max_label = label
				max_prior = prior
		return max_label

	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			label_probs = dict()
			for label, prior in self.priors.items():
				label_probs[label] = prior
				for i, x in enumerate(dct['features']):
					if x==1:
						label_probs[label] *= self.conditionalProb(i, label)
			max_label = None
			max_val = 0
			lst = label_probs.values()			
			for label, probs in label_probs.items():
				if probs > max_val or max_label is None:
					max_label = label
					max_val = probs
			predictions.append(max_label)
		return predictions

def ZeroOneLoss(true_labels, predictions):
	loss = 0
	assert len(true_labels) == len(predictions)
	for i, x in enumerate(true_labels):
		if x != predictions[i]:
			loss += 1
	return loss/len(predictions)

def runNBCExperiment(percentage, numFeatures):
	Selector = selector(os.getcwd() + '/yelp_data.csv')
	train_data, test_data = Selector.read(percentage)
	TestPreProcessor = preprocessor()
	train_data = TestPreProcessor.process(train_data)
	test_data = TestPreProcessor.process(test_data)
	TestTransformer = transformer(train_data, numFeatures)
	train_data = TestTransformer.transform()
	test_data = TestTransformer.transform_test(test_data)
	NBModel = NaiveBayesModel()
	NBModel.train(train_data)
	true_labels = [dct['label'] for dct in test_data]
	NBpredictions = NBModel.predict(test_data)
	defaultPrediction = NBModel.defaultPrediction()
	Dpredictions = \
		[defaultPrediction for x in range(0, len(test_data))]
	return true_labels, Dpredictions, NBpredictions

def exploreTrainingSize(percentageList):
	Dmeans = []
	Dstds = []
	NBmeans = []
	NBstds = []
	for percentage in percentageList:
		Dlosses = []
		NBlosses = []
		for run in range(0, 10):
			true_labels, Dpredictions, NBpredictions = \
				runNBCExperiment(percentage, 500)
			NBloss = ZeroOneLoss(true_labels, NBpredictions)
			Dloss = ZeroOneLoss(true_labels, Dpredictions)
			Dlosses.append(Dloss)
			NBlosses.append(NBloss)
		Dmeans.append(np.mean(Dlosses))
		Dstds.append(np.std(Dlosses))
		NBmeans.append(np.mean(NBlosses))
		NBstds.append(np.mean(NBlosses))
	return Dmeans, Dstds, NBmeans, NBstds

def exploreFeatureSize(numFeaturesList):
	Dmeans = []
	Dstds = []
	NBmeans = []
	NBstds = []
	for numFeatures in numFeaturesList:
		Dlosses = []
		NBlosses = []
		for run in range(0, 10):
			true_labels, Dpredictions, NBpredictions = \
				runNBCExperiment(50, numFeatures)
			NBloss = ZeroOneLoss(true_labels, NBpredictions)
			Dloss = ZeroOneLoss(true_labels, Dpredictions)
			Dlosses.append(Dloss)
			NBlosses.append(NBloss)
		Dmeans.append(np.mean(Dlosses))
		Dstds.append(np.std(Dlosses))
		NBmeans.append(np.mean(NBlosses))
		NBstds.append(np.mean(NBlosses))
	return Dmeans, Dstds, NBmeans, NBstds

def plotErrorBar(X, Y, Y_err, Y2, Y2_err, title, xlabel, ylabel):
	fig, ax = plt.subplots()	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Zero-One Loss')
	ax.errorbar(X, Y, yerr=Y_err, label='Default Baseline Classifier')
	ax.set_xlim([0,max(X) + 10])
	ax.errorbar(X, Y2, yerr=Y2_err, label='Naive Bayesian Classifier')
	ax.legend()
	plt.savefig(title + '.pdf')  
	plt.clf()
	plt.cla()
	plt.close()

def generateReports():
	print("Generating reports...")
	numFeaturesList = [10, 50, 250, 500, 1000, 4000]
	percentageList = [1, 5, 10, 20, 50, 90]
	Dmeans, Dstds, NBmeans, NBstds = exploreTrainingSize(percentageList)
	xlabel = 'Training-set size as percentage of data'
	title = 'Training-set Size vs Zero-one Loss'
	ylabel = 'Zero-one Loss'
	plotErrorBar(percentageList, Dmeans, Dstds, NBmeans, NBstds, title, xlabel, ylabel)
	Dmeans, Dstds, NBmeans, NBstds = exploreFeatureSize(numFeaturesList)
	xlabel = 'Feature size'
	title = 'Feature size vs Zero-one Loss'
	ylabel = 'Zero-one Loss'
	plotErrorBar(numFeaturesList, Dmeans, Dstds, NBmeans, NBstds, title, xlabel, ylabel)

def reportScoresOnScreen(train_file, test_file):
	Selector = selector(train_file)
	train_data, _dummy = Selector.read(100)
	TestPreProcessor = preprocessor()
	train_data = TestPreProcessor.process(train_data)
	test_data, _dummy = selector(test_file).read(100)
	test_data = TestPreProcessor.process(test_data)
	TestTransformer = transformer(train_data, 500)
	train_data = TestTransformer.transform()
	test_data = TestTransformer.transform_test(test_data)
	NBModel = NaiveBayesModel()
	NBModel.train(train_data)
	true_labels = [dct['label'] for dct in test_data]
	NBpredictions = NBModel.predict(test_data)
	loss = ZeroOneLoss(true_labels, NBpredictions)
	print("ZERO-ONE-LOSS", loss)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		generateReports()
	else:
		reportScoresOnScreen(sys.argv[1], sys.argv[2])