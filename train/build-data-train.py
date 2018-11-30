#-*- coding: utf-8 -*-

from tqdm import tqdm    # Nice progressbar

import random
from glob import glob
import re
listfile=glob('./best/*/*.txt') # Best
from pythainlp.tokenize.tcc import tcc
def cut(word):
    return tcc(word, sep="ii/ii").split('ii/ii')

def lines_from_file(filepath):
    '''
        read in a file as a list of words, ex:
        [['กฎหมาย', 'กับ', 'การ', 'เบียดบัง', 'คน', 'จน'], 
         ['จาก', 'ต้นฉบับ', 'เรื่อง', 'คน', 'จน', 'ภาย', 'ใต้', 'กฎหมาย'], 
         ['ไพสิฐ พาณิชย์กุล']]
    '''
    lines = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for s in f: # s is the line string
            if s and (len(s) > 0):
                tokens = re.split('[|]', s.strip())[:-1]
                words = [] # words in a line
                for t in tokens:
                    t = re.sub('<[^>]*>', '', t)
                    words += [t]
                lines.append(words) #+= [words]
    return lines

linesall = []
for i in listfile:
    linesall.append(lines_from_file(i))
t=[]

for i in linesall:
    for j in i:
        t.append(j)
t=[i for i in t if i!=[]]
corpus=t
crplen = len(corpus)
print('Number of sentences = {}'.format(crplen))
def lines_from_file2(filepath):
    '''
        read in a file as a list of words, ex:
        [['กฎหมาย', 'กับ', 'การ', 'เบียดบัง', 'คน', 'จน'], 
         ['จาก', 'ต้นฉบับ', 'เรื่อง', 'คน', 'จน', 'ภาย', 'ใต้', 'กฎหมาย'], 
         ['ไพสิฐ พาณิชย์กุล']]
    '''
    lines = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for s in f: # s is the line string
            if s and (len(s) > 0):
                tokens = re.split('[|]', s.strip().lower())[:-1]
                words = [] # words in a line
                for t in tokens:
                    t = re.sub('<[^>]*>', '', t)
                    words += [cut(t)]
                lines.append(words) #+= [words]
    return lines

linesall2 = []
for i in listfile:
    linesall2.append(lines_from_file2(i))
t2=[]

for i in linesall2:
    for j in i:
        t2.append(j)

all_subword = set()
for sent in t2:
    for subword in sent:
        all_subword.update(set(subword))

idx2subword = [None] + sorted(all_subword)
print(idx2subword)

no_subword = len(idx2subword)
print('Number of subword = {}'.format(no_subword))
subword2idx = {}
for (idx, subword) in enumerate(idx2subword):
    subword2idx[subword] = idx
print(subword2idx)
def str2idxseq(seq):
    idxseq = []
    for char in cut(seq):
        char = char.lower()
        if char in subword2idx:
            idxseq.append(subword2idx[char])
        else:
            idxseq.append(subword2idx[None])
    return idxseq

def idxseq2str(idxseq):
    charseq = []
    for idx in idxseq:
        if idx < len(idx2subword):
            charseq.append(idx2subword[idx])
        else:
            charseq.append(' ')
    return charseq

def sent2data(sent):
    charidxs = []
    wordbrks = []
    for charseq in sent:
        idxs = str2idxseq(charseq)
        charidxs.extend(idxs)
        wordbrks.extend((len(idxs) - 1) * [False] + [True])
    return (charidxs, wordbrks)
  

def corpus2dataset(corpus):
    dataset = []
    for sent in corpus:
        charidxs, wordbrks = sent2data(sent)
        dataset.append((charidxs, wordbrks))
    return dataset

training_len = int(crplen * 0.9)           # 90% for training, 10% for testing
training_set = corpus2dataset(corpus[: training_len])
testing_set = corpus2dataset(corpus[training_len :])
print('Size of training set = {}'.format(len(training_set)))
print('Size of testing set = {}'.format(len(testing_set)))

import pickle
filehandler = open("subword2idx.lab6", 'wb')
pickle.dump(subword2idx, filehandler)
filehandler.close()
filehandler = open("idx2subword.lab6", 'wb')
pickle.dump(idx2subword, filehandler)
filehandler.close()

filehandler = open("training_set.lab6", 'wb')
pickle.dump(training_set, filehandler)
filehandler.close()
filehandler = open("testing_set.lab6", 'wb')
pickle.dump(testing_set, filehandler)
filehandler.close()
