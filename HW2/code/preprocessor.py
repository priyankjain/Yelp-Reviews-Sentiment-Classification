###############################################################################
#Author: Priyank Jain (@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
from selector import selector
import os
import string
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

if __name__ == "__main__":
	Selector = selector(os.getcwd() + '/../data/yelp_data.csv')
	train_data, test_data = Selector.read(10)
	TestPreProcessor = preprocessor(data)
	train_data = TestPreProcessor.process(train_data)
	test_data = TestPreProcessor.process(test_data)
	print(train_data, test_data)
	print(len(train_data), len(test_data))