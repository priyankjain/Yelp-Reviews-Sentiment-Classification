###############################################################################
#Author: Priyank Jain (@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
import os.path
import codecs
from random import shuffle
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

if __name__ == '__main__':
	TestSelector = selector(os.getcwd() + '/../data/yelp_data.csv')
	train_data, test_data = TestSelector.read(10)
	print(train_data, test_data)
	print(len(train_data), len(test_data))