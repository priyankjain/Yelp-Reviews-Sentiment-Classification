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
import copy
from collections import OrderedDict
import random
from pprint import pprint
from scipy.stats import ttest_rel
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
		#shuffle(self.data)
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
	def __init__(self, data, featureCount, stopwordsCount=100,\
		binary = True):
		self.data = data
		self.featureCount = featureCount
		self.stopwordsCount = stopwordsCount
		self.feature_ordered_set = None
		self.binary = True

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
		sorted_hist = dict(sorted_hist[:self.featureCount])
		feature_set = set(sorted_hist.keys())
		feature_ordered_set = OrderedDict([(feature, 0) for feature in feature_set])
		for dct in self.data:
			featureDict = OrderedDict(\
				[(feature, 0) for feature in feature_ordered_set.keys()])
			for word in dct['text']:
				if word in feature_set:
					if self.binary == True:
						featureDict[word] = 1
					else:
						if featureDict[word] < 2:
							featureDict[word] += 1
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
					if self.binary == True:
						featureDict[word] = 1
					else:
						if featureDict[word] < 2:
							featureDict[word] += 1
			dct['features'] = list(featureDict.values())
			dct.pop('text', None)
		return test_data

class NaiveBayesModel(object):
	def __init__(self, smoothing=True, numLabels=2, binary=True):
		self.smoothing = smoothing
		self.numLabels = numLabels
		self.priors = dict()
		self.conditionals = dict()
		self.binary = binary
		self.featureValCount = 2
		if not self.binary:
			self.featureValCount = 3

	def train(self, data):
		for dct in data:
			self.priors.setdefault(dct['label'], 0)
			self.priors[dct['label']] += 1
			featuresCount = len(dct['features'])
			self.conditionals.setdefault(dct['label'], \
				[[0 for i in range(0, 3)]
				 for x in range(0, featuresCount)])
			features = copy.deepcopy(dct['features'])
			features = [[1,0,0] if x==0 else [0,1,0] if x==1\
			else [0,0,1] for x in features]
			self.conditionals[dct['label']]  = [\
			[x[i]+y[i] for i in range(0,3)] for x,y in \
				zip(self.conditionals[dct['label']], features)]

	def priorProb(self, y):
		return self.priors[y]/sum(self.priors)

	def conditionalProb(self, x,y,xval):
		return (self.conditionals[y][x][xval] + 1)/(\
			self.priors[y]+self.featureValCount)

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
				label_probs[label] = math.log(prior)
				for i, x in enumerate(dct['features']):
					label_probs[label] += math.log(self.conditionalProb(i, label, x))
			max_label = None
			max_val = 0
			for label, probs in label_probs.items():
				if probs > max_val or max_label is None:
					max_label = label
					max_val = probs
			predictions.append(max_label)
		return predictions

class LogisticRegression(object):
	def __init__(self, learning_rate=0.01, \
		reg_lambda=0.01, max_iter=100, tol=1e-6):
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.max_iter = max_iter
		self.tol = tol

	@staticmethod
	def logistic(theta):
		return 1/(1+math.exp(-theta))

	@staticmethod
	def round(x):
		if x>= 0.5:
			return 1
		return 0

	def train(self, data):
		featuresCount = len(data[0]['features'])
		featuresCount += 1 # Bias Term
		self.weights = [0 for i in range(0, featuresCount)]
		for dct in data:
			dct['features'].append(1) # Add a constant feature 1 -> bias
		for iter_num in range(1, self.max_iter+1):
			predictions = [LogisticRegression.logistic(\
					np.dot(self.weights, x['features']))\
			for x in data]
			diff = [x['label']-y for x, y in zip(data, predictions)]
			gradient = [np.dot(diff, [x['features'][i] \
				for x in data])\
			- self.reg_lambda*self.weights[i] \
			for i in range(0, featuresCount)]
			ascent = [x*self.learning_rate for x in gradient]
			if np.linalg.norm(ascent) < self.tol:
				break
			self.weights = [x+y for x,y in \
				zip(self.weights, ascent)]

	def predict(self, test_data):
		for dct in test_data:
			dct['features'].append(1) # Add a constant feature 1 -> bias
		predictions = [LogisticRegression.round(\
				LogisticRegression.logistic(\
					np.dot(self.weights, x['features'])))\
			for x in test_data]
		return predictions

class SVM(object):
	def __init__(self, learning_rate=0.5, \
		reg_lambda=0.01, max_iter=100, tol=1e-6):
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda
		self.max_iter = max_iter
		self.tol = tol

	@staticmethod
	def round(x):
		if x>=0:
			return 1
		return -1

	def train(self, data):
		featuresCount = len(data[0]['features'])
		featuresCount += 1
		self.weights = [0 for i in range(0, featuresCount)]
		for dct in data:
			dct['features'].append(1) #Append bias related feature
			if dct['label'] == 0:
				dct['label'] = -1
		for iter_num in range(1, self.max_iter+1):
			predictions = [np.dot(\
				self.weights, x['features']) for x in data]
			prod = [x['label']*y for x,y in zip(data,predictions)]
			delta = [[dct['label']*x for x in dct['features']] for dct in data]
			delta = [x if y<1 else [0 for i in range(0, featuresCount)]\
			 for x,y in zip(delta, prod)]			
			delta = list(np.transpose(np.array(delta)))
			delta = [sum(x)/len(data) for x in delta]
			gradient = [self.reg_lambda*x - y for x, y in \
				zip(self.weights,delta)]
			descent = [self.learning_rate*x for x in gradient]
			if np.linalg.norm(descent) < self.tol:
				break
			self.weights = [x - y for x,y in zip(self.weights, descent)]

	def predict(self, test_data):
		for dct in test_data:
			dct['features'].append(1) # Add a constant feature 1 -> bias
		predictions = [SVM.round(\
				np.dot(self.weights, x['features']))\
			for x in test_data]
		predictions = [1 if x==1 else 0 for x in predictions]
		return predictions

def ZeroOneLoss(true_labels, predictions):
	loss = 0
	assert len(true_labels) == len(predictions)
	for i, x in enumerate(true_labels):
		if x != predictions[i]:
			loss += 1
	return loss/len(predictions)

def plotErrorBar(X, Y, Y_err, Y2, Y2_err, Y3, Y3_err, title, xlabel, ylabel):
	fig, ax = plt.subplots()	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Zero-One Loss')
	ax.errorbar(X, Y, yerr=Y_err, label='Naive Bayesian Classifier')
	ax.set_xlim([0,max(X) + 10])
	ax.errorbar(X, Y2, yerr=Y2_err, label='Logistic Regression')
	ax.errorbar(X, Y3, yerr=Y3_err, label='Support Vector Machine')
	ax.legend()
	plt.savefig(title + '.pdf')  
	plt.clf()
	plt.cla()
	plt.close()

def generateReports():
	NBCMeansList = []
	SVMMeansList = []
	LRMeansList = []
	for binary in [True, False]:
		text = None
		if binary:
			text = "binary features"
		else:
			text = "features with three values"
		print("Generating reports for {}".format(text))
		Selector = selector(os.getcwd() + '/yelp_data.csv')
		data, test_data = Selector.read(100)
		TestPreProcessor = preprocessor()
		data = TestPreProcessor.process(data)
		D = len(data)
		fold_size = int(D/10)
		training_set_sizes = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		training_set_sizes = [int(D*x) for x in training_set_sizes]
		NBCmeans = []
		NBCstds = []
		SVMmeans = []
		SVMstds = []
		LRmeans = []
		LRstds = []
		for tss in training_set_sizes:
			NBCLosses = []
			SVMLosses = []
			LRLosses = []
			for fold in range(0, 10):
				dcopy = copy.deepcopy(data)
				test_data = dcopy[fold*fold_size: (fold+1)*fold_size]
				union = dcopy[0:fold*fold_size] + dcopy[(fold+1)*fold_size:]
				shuffle(union)
				train_data = union[:tss]
				TestTransformer = transformer(train_data, \
					4000, binary=binary)
				train_data = TestTransformer.transform()
				test_data = TestTransformer.transform_test(test_data)
				NBModel = NaiveBayesModel(binary=binary)
				NBModel.train(train_data)
				true_labels = [dct['label'] for dct in test_data]
				NBpredictions = NBModel.predict(test_data)
				loss = ZeroOneLoss(true_labels, NBpredictions)
				NBCLosses.append(loss)
				print('TSS', tss)
				lr = LogisticRegression()
				lr.train(train_data)
				LRpredictions = lr.predict(test_data)
				loss = ZeroOneLoss(true_labels, LRpredictions)
				LRLosses.append(loss)
				svm = SVM()
				svm.train(train_data)
				SVMpredictions = svm.predict(test_data)
				loss = ZeroOneLoss(true_labels, SVMpredictions)
				SVMLosses.append(loss)
			NBCmeans.append(np.mean(NBCLosses))
			SVMmeans.append(np.mean(SVMLosses))
			LRmeans.append(np.mean(LRLosses))
			NBCstds.append(np.std(NBCLosses)/math.sqrt(10))
			SVMstds.append(np.std(SVMLosses)/math.sqrt(10))
			LRstds.append(np.std(LRLosses)/math.sqrt(10))
		xlabel = 'Training-set size'
		title = 'Training-set Size vs Zero-one Loss'
		if binary:
			title += ' for binary features'
		else:
			title += ' for features with three values'
		ylabel = 'Zero-one Loss'
		plotErrorBar(training_set_sizes, NBCmeans, NBCstds, \
			LRmeans, LRstds, SVMmeans, SVMstds, title, xlabel, ylabel)
		LRMeansList.append(LRmeans)
		SVMMeansList.append(SVMmeans)
		NBCMeansList.append(NBCmeans)
	return NBCMeansList, LRMeansList, SVMMeansList

def reportScoresOnScreen(train_file, test_file, modelIdx):
	Selector = selector(train_file)
	train_data, _dummy = Selector.read(100)
	TestPreProcessor = preprocessor()
	train_data = TestPreProcessor.process(train_data)
	test_data, _dummy = selector(test_file).read(100)
	test_data = TestPreProcessor.process(test_data)
	TestTransformer = transformer(train_data, 4000)
	train_data = TestTransformer.transform()
	test_data = TestTransformer.transform_test(test_data)
	model = None
	if modelIdx=='1':
		model = LogisticRegression()
	elif modelIdx == '2':
		model = SVM()
	model.train(train_data)
	true_labels = [dct['label'] for dct in test_data]
	predictions = model.predict(test_data)
	loss = ZeroOneLoss(true_labels, predictions)
	if modelIdx=='1':
		print("ZERO-ONE-LOSS-LR", loss)
	elif modelIdx == '2':
		print("ZERO-ONE-LOSS-SVM", loss)

def hypothesisTesting(NBCmeans, LRmeans, SVMmeans):
	print('Means with NBC with binary features', NBCmeans[0])
	print('Means with NBC with features with 3 values', NBCmeans[1])
	print('Means with LR with binary features', LRmeans[0])
	print('Means with LR with features with 3 values', LRmeans[1])
	print('Means with SVM with binary features', SVMmeans[0])
	print('Means with SVM with features with 3 values', SVMmeans[1])
	#NBC versus LR
	t, p = ttest_rel(NBCmeans[0], LRmeans[0])
	print("ttest_rel for NBC vs LR: t = %g  p = %g" % (t, p))
	#NBC versus SVM
	t, p = ttest_rel(NBCmeans[0], SVMmeans[0])
	print("ttest_rel for NBC vs SVM: t = %g  p = %g" % (t, p))
	#SVM versus LR
	t, p = ttest_rel(SVMmeans[0], LRmeans[0])
	print("ttest_rel for SVM vs LR: t = %g  p = %g" % (t, p))
	#NBC vs NBC
	t, p = ttest_rel(NBCmeans[0], NBCmeans[1])
	print("ttest_rel for NBC with binary-valued features\
vs NBC with features with three values\
: t = %g  p = %g" % (t, p))
	#LR vs LR
	t, p = ttest_rel(LRmeans[0], LRmeans[1])
	print("ttest_rel for LR with binary-valued features\
vs LR with features with three values\
: t = %g  p = %g" % (t, p))
	#SVM vs SVM
	t, p = ttest_rel(SVMmeans[0], SVMmeans[1])
	print("ttest_rel for SVM with binary-valued features\
vs SVM with features with three values\
: t = %g  p = %g" % (t, p))

if __name__ == "__main__":
	if len(sys.argv) != 4:
		NBCmeans, LRmeans, SVMmeans = generateReports()
		hypothesisTesting(NBCmeans, LRmeans, SVMmeans)
	else:
		reportScoresOnScreen(sys.argv[1], sys.argv[2], sys.argv[3])
	

	