###############################################################################
#Author: Priyank Jain (@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
import os
from selector import selector
from preprocessor import preprocessor
import operator
from collections import OrderedDict
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
		sorted_hist = sorted(hist.items(), key = operator.itemgetter(1), reverse=True)
		sorted_hist = sorted_hist[self.stopwordsCount:]
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

if __name__ == "__main__":
	Selector = selector(os.getcwd() + '/../data/yelp_data.csv')
	train_data, test_data = Selector.read(10)
	TestPreProcessor = preprocessor()
	train_data = TestPreProcessor.process(train_data)
	test_data = TestPreProcessor.process(test_data)
	TestTransformer = transformer(train_data, 500)
	train_data = TestTransformer.transform()
	test_data = TestTransformer.transform_test(test_data)
	print(train_data, test_data)
	print(len(train_data), len(test_data))