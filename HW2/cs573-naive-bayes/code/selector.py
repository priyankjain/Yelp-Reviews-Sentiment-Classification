###############################################################################
#Author: Priyank Jain(@priyankjain)
#Email: jain206@purdue.edu
###############################################################################
import os.path
import codecs
class selector(object):
	def __init__(self, file, percentage=100):
		if not os.path.exists(file):
			raise FileNotFoundError(file + ' does not exist')
		if not isinstance(percentage, (float, int)) or percentage < 0 \
			or percentage > 100:
			raise ValueError('Percentage is invalid')
		self.file = file
		self.percentage = percentage

	def read(self):
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
		print(self.data)

	

if __name__ == '__main__':
	TestSelector = selector(os.getcwd() + '/../data/yelp_data.csv', 10)
	TestSelector.read()