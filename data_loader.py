# load and clean data from data folder
import os.path
import numpy as np

from collections import Counter()

PATH = 'data'
START_CHAR = '^'
STOP_CHAR = '$'
UNK_CHAR = '|'

class Loader:
	def __init__(self, batch_size):
		train_f = open(path.join(PATH, 'ptb.train.txt'), 'r')
		test_f = open(path.join(PATH, 'ptb.test.txt'), 'r')
		valid_f = open(path.join(PATH, 'ptb.valid.txt'), 'r')

		self.train = [line.split(' ')[1:-1]) for line in train_f]
		self.test = [line.split(' ')[1:-1]) for line in test_f]
		self.valid = [line.split(' ')[1:-1]) for line in valid_f]
		
		train_f.close()
		test_f.close()
		valid_f.close()

		self.seq_len = 35
		self.batch_size = batch_size
		self.batch_ptr = 0
		self.train_samples = len(self.train)

		# get/set vocab information
		self.max_word_len = 0
		char_counts = Counter()
		word_counts = Counter()
		for line in self.train:
			# replace unk with special char
			for word in line:
				self.max_word_len = max(self.max_word_len, len(word))
				word_counts[word] += word
				char_counts[START_CHAR] += 1
				char_counts[STOP_CHAR] += 1
				for char in word:
					char_counts[char] += char

		self.max_word_len += 2 # add start and stop char
		self.char_vocab_size = len(char_counts)
		self.word_vocab_size = len(word_counts)

		# id 0 is reserved for null output
		self.char2id = dict()
		self.id2char = char_counts.most_common()
		for char in self.id2char:
			self.char2id[char] = len(self.char2id) + 1

		self.word2id = dict()
		self.id2word = word_counts.most_common()
		for word in self.id2word:
			self.word2id[word] = len(self.id2word) + 1


	def next_batch():
		if self.batch_ptr >= self.train_samples:
			return None

		this_batch_size = min(self.batch_size, self.train_samples - self.batch_ptr - 1)
		targets = np.zeros(this_batch_size, self.seq_len)
		features = np.zeros(this_batch_size, self.seq_len, self.max_word_len)
		# process batch, preprocess batch if not preprocessed?
		return targets, features


	# return starting batch of next epoch
	def next_epoch():
		self.batch_ptr = 0
		return self.next_batch()


	def sentence2tensor(sentence):
		# convert sentence to encoded matrix
		if type(sentence) is str:
			sentence = str.split(' ')

		targets = np.zeros([len(sentence)])
		features = np.zeros([len(sentence), self.max_word_len])
		for i, word in enumerate(sentence):
			if word == '<unk>':
				targets[i] = self.word2id[UNK_CHAR]
				features[i, 0] = self.char2id[START_CHAR]
				features[i, 1] = self.char2id[UNK_CHAR]
				features[i, 2] = self.char2id[STOP_CHAR]
			else:
				targets[i] = self.word2id[word]
				features[i, 0] = self.char2id[START_CHAR]
				features[i, len(word) + 1] = self.char2id[STOP_CHAR]
				for j, char in enumerate(word):
					features[i, j] = self.char2id[char]

		return targets, features

