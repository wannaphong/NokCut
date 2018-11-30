#-*- coding: utf-8 -*-
import torch as T
import torch.nn as N
import torch.optim as O

from tqdm import tqdm    # Nice progressbar
import random
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
import pickle
filehandler = open("training_set.lab6", 'rb')
training_set = pickle.load(filehandler)
filehandler.close()
filehandler = open("testing_set.lab6", 'rb')
testing_set = pickle.load(filehandler)
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
            batch_first=True, bidirectional=True#, dropout=0.1
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
wordseg_model = WordsegModel(dim_charvec=300, dim_trans=64*2, no_layers=2)#.cuda()

def wordbrks2brkvec(wordbrks):
    brkvec = T.LongTensor(len(wordbrks)).to(device)
    for i in range(len(wordbrks)):
        if wordbrks[i]: brkvec[i] = 0
        else: brkvec[i] = 1
    return brkvec

def test_model(wordseg_model, testing_data):
    no_correct = 0.0
    no_goldbrks = 0.0
    no_predbrks = 0.0

    for (charidxs, gold_wordbrks) in tqdm(testing_data):
        pred_brkvecs = wordseg_model(charidxs).to(device)
        pred_wordbrks = []
        for i in range(len(charidxs)):
            pred_wordbrk = (pred_brkvecs[i][0] > pred_brkvecs[i][1])
            pred_wordbrks.append(pred_wordbrk)
        
        for i in range(len(charidxs)):
            if gold_wordbrks[i] and gold_wordbrks[i] == pred_wordbrks[i]:
                no_correct += 1.0
            if gold_wordbrks[i]:
                no_goldbrks += 1.0
            if pred_wordbrks[i]:
                no_predbrks += 1.0

    precision = 100 * no_correct / no_predbrks
    recall = 100 * no_correct / no_goldbrks
    f1 = 2 * precision * recall / (precision + recall)
    
    print('\nPrecision = {}'.format(precision))
    print('Recall = {}'.format(recall))
    print('F1 = {}'.format(f1))
    return f1

def train_model(wordseg_model, training_data, epochs, loss_fn, optimizer):
    no_samples = len(training_data)
    loss_history = []
    
    for i in range(epochs):
        total_loss = 0.0
        val_loss = 0.0
        random.shuffle(training_data)
        wordseg_model.train()
        for (charidxs, wordbrks) in tqdm(training_data):                
            try:
                pred_brkvecs = wordseg_model(charidxs).to(device)       # Perform prediction
                gold_brkvec = wordbrks2brkvec(wordbrks).to(device)      # Gold standard
                loss = loss_fn(pred_brkvecs, gold_brkvec)
                total_loss += loss.item()
                optimizer.zero_grad()      # Clear gradient cache
                loss.backward()            # Perform backpropagation
                optimizer.step()           # Update the model parameters
            except:
                print(charidxs)
                pass
        loss_history.append(total_loss / no_samples)
        model.eval()
        for (charidxs, wordbrks) in tqdm(training_data):                
            try:
                pred_brkvecs = wordseg_model(charidxs).to(device)       # Perform prediction
                gold_brkvec = wordbrks2brkvec(wordbrks).to(device)      # Gold standard
                loss = loss_fn(pred_brkvecs, gold_brkvec)
                val_loss += loss.item()
            except:
                print(charidxs)
                pass
        print("Ep : "+str(i+1))
        print("Training Loss : "+str(total_loss / no_samples))
        print("val Loss : "+str(val_loss / no_samples))
        f1=test_model(wordseg_model, testing_set)
        print("F1 : "+str(f1))
        T.save(wordseg_model.state_dict(), "lab6.ep"+str(i+1)+".loss_"+str(total_loss / no_samples)+".val_"+str(val_loss / no_samples)+".f1_"+str(f1)+".model")
        
epochs = 30
# epochs = 20      # Only if you can wait
learning_rate = 0.0001
wordseg_model.to(device)

loss_fn = N.NLLLoss()
optimizer = O.Adam(wordseg_model.parameters(), lr=learning_rate)
train_model(wordseg_model, training_set, epochs, loss_fn, optimizer)
