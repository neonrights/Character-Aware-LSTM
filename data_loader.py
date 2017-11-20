# load and clean data from data folder
import numpy as np
import pickle

from collections import Counter
from os import path

PATH = 'data'
VOCAB = 'vocab.pkl'
WORDS = 'words.pkl'
CHARS = 'chars.pkl'
START_CHAR = '{'
STOP_CHAR = '}'
UNK_CHAR = '|'
SEQUENCE_LENGTH = 35

def save(filename, obj):
    with open(filename, 'w+') as f:
        pickle.dump(obj, f)


def load(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)


class Loader:
    def __init__(self, batch_size):
        vocab_file = path.join(PATH, VOCAB)
        word_file = path.join(PATH, WORDS)
        char_file = path.join(PATH, CHARS)

        if not path.exists(vocab_file) or not path.exists(word_file) or not path.exists(char_file):
            print("failed to processed data, running preprocessor")
            raw_data = [path.join(PATH, 'ptb.train.txt'),
                        path.join(PATH, 'ptb.valid.txt'),
                        path.join(PATH, 'ptb.test.txt')]
            self.preprocess(raw_data, vocab_file, word_file, char_file)

        self.char2id, self.id2char, self.word2id, self.id2word = load(vocab_file)         
        self.word_tensors = load(word_file)
        self.char_tensors = load(char_file)

        self.batch_size = batch_size
        self.seq_len = SEQUENCE_LENGTH
        self.word_vocab_size = len(self.id2word)
        self.char_vocab_size = len(self.id2char)
        self.max_word_len = self.char_tensors[0].shape[1]
        self.batch_ptrs = [0, 0, 0]
        
        self.batch_count = [0, 0, 0]
        # reshape tensors to be samples x seq_len x (1 or max_word_len)

        for i, (word_tensor, char_tensor) in enumerate(zip(self.word_tensors, self.char_tensors)):
            # get rid of offset
            offset = word_tensor.shape[0] % (self.batch_size * self.seq_len)
            word_tensor = word_tensor[:-offset]
            char_tensor = char_tensor[:-offset, :]

            word_tensor = word_tensor.reshape([-1, self.seq_len])
            char_tensor = char_tensor.reshape([-1, self.seq_len, self.max_word_len])

            self.batch_count[i] = word_tensor.shape[0] / self.batch_size
            self.word_tensors[i] = np.split(word_tensor, self.batch_count[i])
            self.char_tensors[i] = np.split(char_tensor, self.batch_count[i])

        print("finished processing data sets. batch size: %d" % self.batch_size)

            
    def preprocess(self, raw_data, vocab_file, word_file, char_file):
        samples = list()
        max_word_len = 0
        for data_file in raw_data:
            with open(data_file, 'r') as f:
                # get/set vocab information
                words = 0
                for line in f:
                    line = line.replace(UNK_CHAR, '')
                    line = line.replace('<unk>', UNK_CHAR)
                    line = line.replace(START_CHAR, '')
                    line = line.replace(STOP_CHAR, '')
                    for word in line.split():
                        words += 1
                        max_word_len = max(max_word_len, len(word))

                samples.append(words)

        max_word_len += 2 # add start and stop char

        char_tensors = list()
        word_tensors = list()

        char2id = {' ': 0, START_CHAR: 1, STOP_CHAR: 2}
        word2id = {'<unk>': 0}
        id2char = [' ', START_CHAR, STOP_CHAR]
        id2word = ['<unk>']

        for i, data_file in enumerate(raw_data):
            char_tensor = np.zeros([samples[i], max_word_len], dtype=np.int32)
            word_tensor = np.zeros(samples[i], dtype=np.int32)
            
            with open(data_file, 'r') as f:
                word_idx = 0
                for line in f:
                    line = line.replace(UNK_CHAR, '')
                    line = line.replace('<unk>', UNK_CHAR)
                    line = line.replace(START_CHAR, '')
                    line = line.replace(STOP_CHAR, '')
                    for word in line.split():
                        if word[0] == '|' and len(word) > 1:
                            word = word[1:]
                            word_tensor[word_idx] = word2id[UNK_CHAR]
                        else:
                            if not word2id.has_key(word):
                                word2id[word] = len(word2id)
                                id2word.append(word)
                            word_tensor[word_idx] = word2id[word]

                        char_tensor[word_idx, 0] = char2id[START_CHAR]
                        for j, char in enumerate(word):
                            if not char2id.has_key(char):
                                char2id[char] = len(char2id)
                                id2char.append(char)
                            char_tensor[word_idx, j+1] = char2id[char]
                        char_tensor[word_idx, len(word)+1] = char2id[STOP_CHAR]

                        word_idx += 1

            char_tensors.append(char_tensor)
            word_tensors.append(word_tensor)

        save(vocab_file, [char2id, id2char, word2id, id2word])
        save(char_file, char_tensors)
        save(word_file, word_tensors)

    # return next batch, return None, None if end of epoch reached
    def next_batch(self, index):
        batch_num = self.batch_ptrs[index]
        if batch_num >= self.batch_count[index]:
            return None, None

        return self.char_tensors[index][batch_num], self.word_tensors[index][batch_num]


    # return starting batch of next epoch
    def next_epoch(self, index):
        self.batch_ptrs[index] = 0


    def sentence2tensor(self, sentence):
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

