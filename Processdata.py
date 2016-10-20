import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re


def build_data_train_test(data_train, data_test, train_ratio=0.8, clean_string=True):
    """
    Loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set
    for i in xrange(data_train.shape[0]):
        line = data_train['review'][i]
        y = data_train['sentiment'][i]
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)

    # Pre-process test data set
    for i in xrange(data_test.shape[0]):
        line = data_test['review'][i]
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(' '.join(rev))
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)
    W[0] = np.zeros(k, dtype=np.float32)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
data_test = pd.read_csv('testData.tsv', sep='\t')

w2v_file = 'GoogleNews-vectors-negative300.bin'

revs, vocab = build_data_train_test(data_train, data_test, train_ratio=0.8, clean_string=True)
max_l = np.max(pd.DataFrame(revs)['num_words'])
print 'data loaded!'
print 'number of sentences: ' + str(len(revs))
print 'vocab size: ' + str(len(vocab))
print 'max sentence length: ' + str(max_l)
print 'loading word2vec vectors...',
w2v = load_bin_vec(w2v_file, vocab)
print 'word2vec loaded!'
print 'num words already in word2vec: ' + str(len(w2v))
add_unknown_words(w2v, vocab)
W, word_idx_map = get_W(w2v)
cPickle.dump([revs, W, word_idx_map, vocab], open('imdb-train-val-test.pickle', 'wb'))
print 'dataset created!'