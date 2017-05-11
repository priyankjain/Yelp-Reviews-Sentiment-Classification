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
import pickle
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
		return self.data, featureDict.keys()

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

class Tree(object):
	def __init__(self):
		self.left = None
		self.right = None
		self.data = None

	def traverse(self, level=0):
		if self.data is not None:
			for i in range(0, level):
				print("\t", end="")
			print(self.data)
			if self.left is not None:
				self.left.traverse(level+1)
			if self.right is not None:
				self.right.traverse(level+1)

class DT(object):
	def __init__(self, attributes, numFeatures=None ,depth_limit=10, example_limit=10):
		self.depth_limit = depth_limit
		self.example_limit = example_limit
		self.root = Tree()
		self.attributes = copy.deepcopy(attributes)
		self.numFeatures = numFeatures
		self.attributeIndices = range(0, len(self.attributes))

	def gini_index(self, data):
		if len(data) == 0:
			return 0
		class_labels = [datum['label'] for datum in data]
		p_1 = sum(class_labels)/len(class_labels)
		p_0 = 1 - p_1
		return 1 - p_0**2 - p_1**2

	def get_split(self, data, depth):
		if depth == self.depth_limit or len(data) < self.example_limit:
			current_node = Tree()
			pos_labels = [datum['label'] for datum in data]
			prediction = 'random'
			if 2*sum(pos_labels) > len(pos_labels):
				prediction = 1
			elif 2*sum(pos_labels) < len(pos_labels):
				prediction = 0 	
			current_node.data = {'nodeType': 'leaf', 'prediction': prediction,\
			'examples': len(data)}
			return current_node
		else:
			pos_labels = [datum['label'] for datum in data]
			if sum(pos_labels) == len(pos_labels):
				current_node = Tree()
				current_node.data = {'nodeType': 'leaf', 'prediction': 1,\
				'examples': len(data)}
				return current_node
			elif sum(pos_labels) == 0:
				current_node = Tree()
				current_node.data = {'nodeType': 'leaf', 'prediction': 0,\
				'examples': len(data)}
				return current_node
			current_gi = self.gini_index(data)
			max_gg = None
			best_attr = None
			best_attr_index = None
			attributes = self.attributes
			attributeIndices = self.attributeIndices
			if self.numFeatures is not None:
				attributeIndices = np.random.choice(self.attributeIndices, \
					size=int(math.sqrt(len(self.attributeIndices))), replace=False)
				attributes = [y for x,y in enumerate(self.attributes) \
				if x in attributeIndices]
			for i, attr in zip(attributeIndices,attributes):
				left_data = [datum for datum in data if datum['features'][i]==0]
				right_data = [datum for datum in data if datum['features'][i]==1]
				left_gi = self.gini_index(left_data)
				right_gi = self.gini_index(right_data)
				gini_gain = current_gi - len(left_data)*left_gi/len(data)\
				- len(right_data)*right_gi/len(data)
				if max_gg is None or gini_gain > max_gg:
					max_gg = gini_gain
					best_attr = attr
					best_attr_index = i
			self.attributes = [x for x in self.attributes if x!=best_attr]
			self.attributeIndices = [x for x in self.attributeIndices if x!= best_attr_index]
			current_node = Tree()
			current_node.data = {'nodeType': 'non-leaf', \
			'name': best_attr, 'index': best_attr_index,\
			'gain': max_gg, 'examples': len(data)}
			left_data = [datum for datum in data if datum['features'][best_attr_index]==0]
			right_data = [datum for datum in data if datum['features'][best_attr_index]==1]
			current_node.left = self.get_split(left_data, depth+1)
			current_node.right = self.get_split(right_data, depth+1)
			return current_node

	def train(self, data):
		self.root = self.get_split(data, 0)
		#self.root.traverse()

	def predictOne(self, data):
		root = self.root
		while root.data['nodeType'] == 'non-leaf':
			if data['features'][root.data['index']]	 == 0:
				root = root.left
			else:
				root = root.right 
		if root.data['prediction'] in [0, 1]:
			return root.data['prediction']
		else:
			return np.random.choice([0,1])

	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			predictions.append(self.predictOne(dct))
		return predictions

class BT(object):
	def __init__(self, featureWords, depth_limit=10, numTrees=50):
		self.DTs = []
		self.numTrees = numTrees
		self.featureWords = featureWords
		self.depth_limit = depth_limit

	def train(self, data):
		for i in range(0, self.numTrees):
			train_data =  np.random.choice(data, size=len(data), \
				replace=True)
			dt = DT(self.featureWords, self.depth_limit)
			dt.train(train_data)
			self.DTs.append(dt)

	def predictOne(self, data):
		predictions = []
		for dt in self.DTs:
			predictions.append(dt.predictOne(data))
		if 2*sum(predictions) > len(predictions):
			return 1
		elif 2*sum(predictions) < len(predictions):
			return 0
		else:
			return np.random.choice([0,1])

	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			predictions.append(self.predictOne(dct))
		return predictions

class RF(object):
	def __init__(self, featureWords, numFeatures, depth_limit=10, numTrees=50):
		self.featureWords = featureWords
		self.numTrees = numTrees
		self.numFeatures = numFeatures
		self.DTs = []
		self.depth_limit = depth_limit

	def train(self, data):
		for i in range(0, self.numTrees):
			train_data =  np.random.choice(data, size=len(data), \
				replace=True)
			dt = DT(self.featureWords, self.numFeatures, self.depth_limit)
			dt.train(train_data)
			self.DTs.append(dt)

	def predictOne(self, data):
		predictions = []
		for dt in self.DTs:
			predictions.append(dt.predictOne(data))
		if 2*sum(predictions) > len(predictions):
			return 1
		elif 2*sum(predictions) < len(predictions):
			return 0
		else:
			return np.random.choice([0,1])

	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			predictions.append(self.predictOne(dct))
		return predictions

class DTW(object):
	def __init__(self, attributes, depth_limit=10, example_limit=10):
		self.depth_limit = depth_limit
		self.example_limit = example_limit
		self.root = Tree()
		self.attributes = copy.deepcopy(attributes)
		self.attributeIndices = range(0, len(self.attributes))

	def gini_index(self, data):
		if len(data) == 0:
			return 0
		class_labels = [datum['label']*datum['w'] for datum in data]
		pos_labels = [x if x>0 else 0 for x in class_labels]
		neg_labels = [-x if x<0 else 0 for x in class_labels]
		p_pos = sum(pos_labels)/(sum(pos_labels)+sum(neg_labels))
		p_neg = 1 - p_pos
		return 1 - p_pos**2 - p_neg**2

	def get_split(self, data, depth):
		if depth == self.depth_limit or len(data) < self.example_limit:
			current_node = Tree()
			labels = [datum['label']*datum['w']  for datum in data]
			pos_labels = [x if x>0 else 0 for x in labels]
			neg_labels = [-x if x<0 else 0 for x in labels]
			prediction = 'random'
			if sum(pos_labels)  > sum(neg_labels):
				prediction = 1
			elif sum(pos_labels) < sum(neg_labels):
				prediction = -1 	
			current_node.data = {'nodeType': 'leaf', 'prediction': prediction,\
			'examples': len(data)}
			return current_node
		else:
			labels = [datum['label']*datum['w']  for datum in data]
			pos_labels = [x if x>0 else 0 for x in labels]
			neg_labels = [-x if x<0 else 0 for x in labels]
			if sum(neg_labels)==0:
				current_node = Tree()
				current_node.data = {'nodeType': 'leaf', 'prediction': 1,\
				'examples': len(data)}
				return current_node
			elif sum(pos_labels) == 0:
				current_node = Tree()
				current_node.data = {'nodeType': 'leaf', 'prediction': -1,\
				'examples': len(data)}
				return current_node
			current_gi = self.gini_index(data)
			max_gg = None
			best_attr = None
			best_attr_index = None
			attributes = self.attributes
			attributeIndices = self.attributeIndices
			for i, attr in zip(attributeIndices,attributes):
				left_data = [datum for datum in data if datum['features'][i]==0]
				right_data = [datum for datum in data if datum['features'][i]==1]
				left_gi = self.gini_index(left_data)
				right_gi = self.gini_index(right_data)
				left_num_ex = [datum['w'] for datum in left_data]
				right_num_ex = [datum['w'] for datum in right_data]
				tot = sum(left_num_ex) + sum(right_num_ex)
				gini_gain = current_gi - sum(left_num_ex)*left_gi/tot\
				- sum(right_num_ex)*right_gi/tot
				if max_gg is None or gini_gain > max_gg:
					max_gg = gini_gain
					best_attr = attr
					best_attr_index = i
			#print(max_gg)
			self.attributes = [x for x in self.attributes if x!=best_attr]
			self.attributeIndices = [x for x in self.attributeIndices if x!= best_attr_index]
			current_node = Tree()
			current_node.data = {'nodeType': 'non-leaf', \
			'name': best_attr, 'index': best_attr_index,\
			'gain': max_gg, 'examples': len(data)}
			left_data = [datum for datum in data if datum['features'][best_attr_index]==0]
			right_data = [datum for datum in data if datum['features'][best_attr_index]==1]
			current_node.left = self.get_split(left_data, depth+1)
			current_node.right = self.get_split(right_data, depth+1)
			return current_node

	def train(self, data):
		self.root = self.get_split(data, 0)
		#self.root.traverse()

	def predictOne(self, data):
		root = self.root
		while root.data['nodeType'] == 'non-leaf':
			if data['features'][root.data['index']]	 == 0:
				root = root.left
			else:
				root = root.right 
		if root.data['prediction'] in [-1, 1]:
			return root.data['prediction']
		else:
			return np.random.choice([-1,1])

	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			predictions.append(self.predictOne(dct))
		return predictions

class BDT(object):
	def __init__(self, featureWords, depth_limit=10, numTrees=50):
		self.featureWords = list(featureWords)
		self.numTrees = numTrees
		self.depth_limit = depth_limit
		self.DTs= []
		self.DTweights = []

	def train(self, data):
		for dct in data:
			dct['w'] = 1/len(data)
			dct['label'] = -1 if dct['label']==0 else 1
		for i in range(0, self.numTrees):
			dtw = DTW(self.featureWords, depth_limit=self.depth_limit)
			dtw.train(data)
			predictions = dtw.predict(data)
			error = sum([x['w'] if x['label']!=y else 0 \
				for x, y in zip(data, predictions)])
			if error == 0:
				error = np.finfo(float).eps
			alpha = 0.5*math.log((1-error)/error)
			self.DTs.append(dtw)
			self.DTweights.append(alpha)
			sum_weights = 0
			for datum, pred in zip(data, predictions):
				datum['w'] *= math.exp(-datum['label']*alpha*pred)
				sum_weights += datum['w']
			for datum in data:
				datum['w'] /= sum_weights
		#print(self.DTweights)
				
	def predict(self, test_data):
		predictions = []
		for dct in test_data:
			prediction = 0
			for t, alpha in zip(self.DTs, self.DTweights):
				t_pred = t.predictOne(dct)
				prediction += alpha*t_pred
			predictions.append(prediction)
		predictions	= [1 if x>=0 else 0 for x in predictions]
		return predictions

def ZeroOneLoss(true_labels, predictions):
	loss = 0
	assert len(true_labels) == len(predictions)
	for i, x in enumerate(true_labels):
		if x != predictions[i]:
			loss += 1
	return loss/len(predictions)

def plotErrorBar(X, means, stds, title, xlabel, exp_no):
	fig, ax = plt.subplots()	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Zero-One Loss')
	if exp_no == 1 or exp_no==2 or exp_no==3 or exp_no==4:
		ax.errorbar(X, means[0], yerr=stds[0], label='Decision Trees')
		ax.set_xlim([0,max(X) + 5])
		ax.errorbar(X, means[1], yerr=stds[1], label='Bagging')
		ax.errorbar(X, means[2], yerr=stds[2], label='Random Forests')
		ax.errorbar(X, means[3], yerr=stds[3], label='Boosting')
		if exp_no != 3 and exp_no!=4:
			ax.errorbar(X, means[4], yerr=stds[4], label='Support Vector Machine')
	ax.legend()
	plt.savefig(title + '.pdf')  
	plt.clf()
	plt.cla()
	plt.close()

def generateReports(training_set_sizes, num_features_list,\
	depth_list, num_trees_list, exp_no):
	models = list()
	if exp_no == 1:
		models = ["DT", "BT", "RF", "BDT", "SVM"]
		#models = ["BDT", "SVM"]
	elif exp_no == 2:
		models = ["DT", "BT", "RF", "BDT", "SVM"]
	elif exp_no == 3 or exp_no == 4:
		models = ["DT", "BT", "RF", "BDT"]
	meansList = [list() for x in models]
	Selector = selector(os.getcwd() + '/yelp_data.csv')
	data, test_data = Selector.read(100)
	TestPreProcessor = preprocessor()
	data = TestPreProcessor.process(data)
	D = len(data)
	fold_size = int(D/10)
	training_set_sizes = [int(D*x) for x in training_set_sizes]
	means = [list() for x in models]
	stds = [list() for x in models]
	hmeans = [list() for x in models]
	for tss in training_set_sizes:
		for numFeatures in num_features_list:
			for depth in depth_list:
				for num_trees in num_trees_list:
					losses = [list() for model in models]
					for fold in range(0, 10):
						dcopy = copy.deepcopy(data)
						test_data = dcopy[fold*fold_size: (fold+1)*fold_size]
						union = dcopy[0:fold*fold_size] + dcopy[(fold+1)*fold_size:]
						shuffle(union)
						train_data = union[:tss]
						TestTransformer = transformer(train_data, \
							numFeatures)
						train_data, featureWords = TestTransformer.transform()
						test_data = TestTransformer.transform_test(test_data)
						featureWords = list(featureWords)
						for model_num, model in enumerate(models):
							print("TSS:", tss, "Features:", numFeatures,\
								"depth:", depth, "num_trees:", num_trees,\
								"model:", model, "fold:", fold)
							train_data = copy.deepcopy(train_data)
							test_data = copy.deepcopy(test_data)
							m = None
							if model=="DT":
								m = DT(featureWords, depth_limit = depth)
							elif model=="BT":
								m = BT(featureWords, depth_limit = depth,\
									numTrees = num_trees)
							elif model=="RF":
								m = RF(featureWords, math.sqrt(len(featureWords)), \
									depth_limit = depth, numTrees = num_trees)
							elif model=="BDT":
								m = BDT(featureWords, depth_limit = depth,\
									numTrees = num_trees)
							else:
								m = SVM()
							m.train(train_data)
							true_labels = [dct['label'] for dct in test_data]
							predictions = m.predict(test_data)
							loss = ZeroOneLoss(true_labels, predictions)
							losses[model_num].append(loss)
					for loss_list in losses:
						print(loss_list)
					for model_num, model in enumerate(models):
						hmeans[model_num].append(losses[model_num])
						means[model_num].append(np.mean(losses[model_num]))
						stds[model_num].append(np.std(losses[model_num])/math.sqrt(10))
	return means, stds, hmeans

def reportScoresOnScreen(train_file, test_file, modelIdx):
	Selector = selector(train_file)
	train_data, _dummy = Selector.read(100)
	TestPreProcessor = preprocessor()
	train_data = TestPreProcessor.process(train_data)
	test_data, _dummy = selector(test_file).read(100)
	test_data = TestPreProcessor.process(test_data)
	TestTransformer = transformer(train_data, 1000)
	train_data, featureWords = TestTransformer.transform()
	test_data = TestTransformer.transform_test(test_data)
	featureWords = list(featureWords)
	if modelIdx=='1':
		model = DT(featureWords)
	elif modelIdx=='2':
		model = BT(featureWords)
	elif modelIdx=='3':
		model = RF(featureWords, math.sqrt(len(featureWords)))
	elif modelIdx=='4':
		model = BDT(featureWords)	
	model.train(train_data)
	true_labels = [dct['label'] for dct in test_data]
	predictions = model.predict(test_data)
	loss = ZeroOneLoss(true_labels, predictions)
	if modelIdx=='1':
		print("ZERO-ONE-LOSS-DT", loss)
	elif modelIdx == '2':
		print("ZERO-ONE-LOSS-BT", loss)
	elif modelIdx == '3':
		print("ZERO-ONE-LOSS-RF", loss)
	elif modelIdx == '4':
		print("ZERO-ONE-LOSS-BDT", loss)

def hypothesisTesting(means, exp_no):
	print(means)
	if exp_no == 1:
		for mIndex, model in enumerate(["DT", "BT", "RF", "BDT"]):
			for tssIndex, tss in enumerate([0.025, 0.05, 0.125, 0.25]):
				t, p = ttest_rel(means[mIndex][tssIndex], means[4][tssIndex])
				print("ttest_rel for %s vs SVM for TSS %g: t = %g  p = %g" \
					% (model, tss, t, p))
	elif exp_no == 2:
		for mIndex, model in enumerate(["DT", "BT", "RF", "BDT"]):
			for numIndex, numFs in enumerate([200, 500, 1000, 1500]):
				t, p = ttest_rel(means[mIndex][numIndex], means[4][numIndex])
				print("ttest_rel for %s vs SVM for number of features %d: t = %g  p = %g" \
					% (model, numFs, t, p))
	elif exp_no == 4:
		for mIndex, model in enumerate(["BT", "RF", "BDT"]):
			for idx, numTrees in enumerate([10, 25, 50, 100]):
				t, p = ttest_rel(means[mIndex+1][idx], means[0][idx])
				print("ttest_rel for %s vs DT for number of trees %d: t = %g p = %g" \
					% (model, numTrees, t, p))
	elif exp_no == 3:
		mlist = ["DT", "BT", "RF", "BDT"]
		for idx1, m1 in enumerate(mlist):
			for idx2 in range(idx1+1, len(mlist)):
				for idx, depth in enumerate([5, 10, 15, 20]):
					t, p = ttest_rel(means[idx1][idx], means[idx2][idx])
					print("ttest_rel for %s vs %s for depth %d: t = %g p = %g"\
						% (m1, mlist[idx2], depth, t, p))

if __name__ == "__main__":
	if len(sys.argv) != 4:
		training_set_sizes = [0.025, 0.05, 0.125, 0.25]
		num_features_list = [200, 500, 1000, 1500]
		depth_list = [5, 10, 15, 20]
		num_trees_list = [10, 25, 50, 100]
		means, stds, hmeans = generateReports(training_set_sizes, [1000], [10], [50], 1)
		pickle.dump(means, open("1.means", "wb"))
		pickle.dump(stds, open("1.stds", "wb"))
		pickle.dump(hmeans, open("1.hmeans", "wb"))
		title = 'Training set size vs Zero-one loss'
		plotErrorBar([50, 100, 250, 500], means, stds, title, 'Training set size', 1)
		hypothesisTesting(hmeans, 1)
		means, stds, hmeans = generateReports([0.25], num_features_list, [10], [50], 2)
		pickle.dump(means, open("2.means", "wb"))
		pickle.dump(stds, open("2.stds", "wb"))
		pickle.dump(hmeans, open("2.hmeans", "wb"))
		title = 'Number of features vs Zero-one loss'
		plotErrorBar(num_features_list, means, stds, title, 'Number of features', 2)
		hypothesisTesting(hmeans, 2)
		means, stds, hmeans = generateReports([0.25], [1000], depth_list, [50], 3)
		pickle.dump(means, open("3.means", "wb"))
		pickle.dump(stds, open("3.stds", "wb"))
		pickle.dump(hmeans, open("3.hmeans", "wb"))
		title = 'Depth of tree vs Zero-one loss'
		plotErrorBar(depth_list, means, stds, title, 'Tree depth', 3)
		hypothesisTesting(hmeans, 3)
		means, stds, hmeans = generateReports([0.25], [1000], [10], num_trees_list, 4)
		pickle.dump(means, open("4.means", "wb"))
		pickle.dump(stds, open("4.stds", "wb"))
		pickle.dump(hmeans, open("4.hmeans", "wb"))
		title = 'Number of trees vs Zero-one loss'
		plotErrorBar(num_trees_list, means, stds, title, 'Number of trees', 4)
		hypothesisTesting(hmeans, 4)
	else:
		reportScoresOnScreen(sys.argv[1], sys.argv[2], sys.argv[3])
	

	