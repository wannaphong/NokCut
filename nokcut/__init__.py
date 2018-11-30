#-*- coding: utf-8 -*-
import pickle
import os
import nokcut
import re
import torch as T
import torch.nn as N
import torch.optim as O
from pythainlp.tokenize.tcc import tcc
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

templates_dir = os.path.join(os.path.dirname(nokcut.__file__))
filehandler = open(os.path.join(templates_dir,"subword2idx.lab6"), 'rb')
subword2idx = pickle.load(filehandler)
filehandler.close()
filehandler = open(os.path.join(templates_dir,"idx2subword.lab6"), 'rb')
idx2subword = pickle.load(filehandler)
filehandler.close()

no_subword=3195

class WordsegModel(N.Module):
    
    def __init__(self, dim_charvec, dim_trans, no_layers):
        super(WordsegModel, self).__init__()
        self._dim_charvec = dim_charvec
        self._dim_trans = dim_trans
        self._no_layers = no_layers
        
        self._charemb = N.Embedding(no_subword, self._dim_charvec).to(device)
        self._rnn = N.GRU(
            self._dim_charvec, self._dim_trans, self._no_layers,
            batch_first=True, bidirectional=True
        ).to(device)
        self._tanh = N.Tanh().to(device)
        self._hidden = N.Linear(2 * self._dim_trans, 2).to(device)    # Predicting two classes: break / no break
        self._log_softmax = N.LogSoftmax(dim=1).to(device)
        
    def forward(self, charidxs):
        try:
            charvecs = self._charemb(T.LongTensor(charidxs).to(device))
            # print('charvecs =\n{}'.format(charvecs))
            ctxvecs, lasthids = self._rnn(charvecs.unsqueeze(0))
            ctxvecs, lasthids = ctxvecs.squeeze(0), lasthids.squeeze(1)
            # print('ctxvecs =\n{}'.format(ctxvecs))
            statevecs = self._hidden(self._tanh(ctxvecs))
            # print('statevecs =\n{}'.format(statevecs))
            brkvecs = self._log_softmax(statevecs)
            # print('brkvecs =\n{}'.format(brkvecs))
            return brkvecs
        except RuntimeError:
            raise RuntimeError(statevecs)
def cut(word):
    return tcc(word, sep="ii/ii").split('ii/ii')

wordseg_model2 = WordsegModel(dim_charvec=300, dim_trans=64*2, no_layers=2)
wordseg_model2.to(device)
wordseg_model2.load_state_dict(T.load(os.path.join(templates_dir,"nok1.model")))
def str2idxseq(seq):
    idxseq = []
    for char in cut(seq):
        char = char.lower()
        if char in subword2idx:
            idxseq.append(subword2idx[char])
        else:
            idxseq.append(subword2idx[None])
    return idxseq

def tokenize(charseq):
    charidxs = str2idxseq(charseq)
    pred_brkvecs = wordseg_model2(charidxs).to(device)
    pred_wordbrks = []
    for i in range(len(charidxs)):
        pred_wordbrk = (pred_brkvecs[i][0] > pred_brkvecs[i][1])
        #print(pred_wordbrk)
        pred_wordbrks.append(pred_wordbrk)
    
    sent = []
    word = []
    begpos = 0
    temp=cut(charseq)
    ss=""
    for i in range(len(pred_wordbrks)):
        if pred_wordbrks[i]==1:
            ss+=temp[i]+"|-|"
        else:
            ss+=temp[i]
        
    ss=ss.split("|-|")
    if ss[len(ss)-1] =='':
        del ss[len(ss)-1]
    return ss
