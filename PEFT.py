# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:28:34 2023

@author: tye
"""
#%% Setup
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor
import os
import numpy as np
import torchtext
import pandas as pd
from tabulate import tabulate
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

rel_path = os.path.dirname(__file__)
rel_path = rel_path.replace("\\", "/")


def printdf(data, nrow = 10):
    df = data.iloc[:nrow]
    print(tabulate(df, tablefmt='psql'))


#%% Preprocessing Methods

batch_size = 512* 2
d_train = Sentiment140_Dataset(rel_path + "/data/train_data.csv")




#d_test = Sentiment140_Dataset(rel_path + "/data/test_data.csv")
train = torch.utils.data.DataLoader(d_train, batch_size = batch_size, shuffle = True)
#test = torch.utils.data.DataLoader(d_test, batch_size = batch_size, shuffle = True)






def one_hot_encode(data, dictionary):
    nrow = len(data)
    ncol = len(dictionary)
    
    
    matrix = np.zeros(shape = (nrow,ncol))
    for i in range(nrow):
        for k in range(len(data[i])):
            col = dictionary[data[i][k]]
            matrix[i,col] = 1.0
    return matrix




import torchtext

def generate_sequences(text_data,#training data
                      maxlen = 50,#maximum length of the embedding sequence
                      max_words = 2000,
                      tokenizer = None):#will only choose consider max_words amount of words for the embedding):
    tok = get_tokenizer("basic_english")
    
    
    
    
    def yield_gen(text):
        for i in text:
            yield tok(i)
            
    tokens_gen = yield_gen(text_data)
   # print(text_data)
    vocab = torchtext.vocab.build_vocab_from_iterator(tokens_gen, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    vocab2text = lambda x: vocab(tok(x))
    vocab_size = len(vocab)

    tokens = torch.empty(size = (len(text_data),maxlen))
    np_text_data = np.array(text_data)
    
    
    
    
    #def tokenize(x):
    #    token = vocab2text(x)
    #    token = torch.tensor(token)
        
        
    #    pad_amount = maxlen - len(token)
    #    if(pad_amount < 0):
    #        token = token.narrow(0,0,maxlen)
    #    else:
    #       token = nn.functional.pad(token, pad = (0,pad_amount))
    #   return token
        
    #tokens_tokenized = torch.tensor.map(tokens, tokenize)    
    
    
    #For each observation, tokenize the token and then 
    #for i in range(len(text_data)):
    #    token = vocab2text(text_data[i])
    #    token = torch.tensor(token)
        
        
    #    pad_amount = maxlen - len(token)
    #    if(pad_amount < 0):
    #        token = token.narrow(0,0,maxlen)
    #    else:
    #        token = nn.functional.pad(token, pad = (0,pad_amount), value=vocab["<unk>"])
    #    tokens[i] = token
    
    
    
   # vocab_size = 0
    for i in range(len(text_data)):
        token = tok(text_data[i])
        indices = [vocab[token[i]] for i in range(len(token))]
        
        pad_amount = maxlen - len(indices)
        if pad_amount < 0:
            indices = indices[:maxlen]
        else:
            indices.extend([vocab["<unk>"]] * pad_amount)
            
        #vocab_size_new = max(vocab_size,max(indices))
      #  if(vocab_size_new != vocab_size):
      #      print(vocab_size_new)
     #       vocab_size = vocab_size_new
        
        tokens[i] = torch.tensor(indices)
        
    
    tok.vocab_size = vocab_size
        
    
    
    return tokens, tok

#def tokens2text:
    


class Sentiment140_Dataset(Dataset):
    def __init__(self, Dir, maxlen = 50, max_words = 500):
        self.dataset = pd.read_csv(Dir, encoding='iso-8859-1', engine = "c").iloc[:,[0,5]]
        #factor from 0 and 4 to either 0 and 1
        self.dataset.iloc[:,0] = self.dataset.iloc[:,0]/4
        
        self.y = np.array(self.dataset.iloc[:,0])
        self.x,self.tokenizer = generate_sequences(np.array(self.dataset.iloc[:,1]),
                                    maxlen,
                                    max_words)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
#%%


#%%

d_train.x.size()

#%%
for i in range(len(d_train.x[:,])):
    if(len(d_train.x[i,]) != 50):
        print(i)




#%%


maxlen = 50
embedding_dim = 64
hidden_dim = 64

class LSTM_Network(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        
        #print("LSTM Layer Vocab size: " + str(vocab_size))
        
        super(LSTM_Network, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to label space
        self.hidden2tag = nn.Linear(hidden_dim,1)
    
    def forward(self, sentence):
        
        print(len(sentence))
       
       
        #print(type(sentence.long()))
        embeds = self.word_embeddings(sentence.long())
        #embeds = torch.transpose(embeds,0,1)
        
        
        #print(embeds.reshape(-1,embedding_dim).size())
        embeds = embeds.view(batch_size, -1, embedding_dim)
        
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :] 

        
        #print(lstm_out.size())
        
        
        
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space)
        return tag_scores
        
model = LSTM_Network(embedding_dim, hidden_dim, d_train.tokenizer.vocab_size, 2)
loss_fn = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr = 0.01)


#loss_fn = nn.BCELoss()  # binary cross entropy
#optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 5    # number of epochs to run
#batch_size = 10  # size of each batch
batches_per_epoch = len(d_train.x) // batch_size
 


model.to(device)

for epoch in range(n_epochs):
    print("Epoch: " + str(epoch + 1))
    i = 0
    for xbatch, ybatch in train:
        #start = i * batch_size
        # take a batch
        #Xbatch = i
        #ybatch = train.y[start:start+batch_size]
        
        
        #ybatch = torch.from_numpy(ybatch)
        
        # forward pass
        ybatch = ybatch.float()
        ybatch = ybatch.unsqueeze(1)
    
        batch_size = len(ybatch)
    
    
        Xbatch = Xbatch.to(device)
        ybatch = ybatch.to(device)
        y_pred = model(Xbatch)
        
        
        loss = loss_fn(y_pred, ybatch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        
        
        predictions = y_pred.argmax(1)
        Accuracy = torch.mean((predictions == ybatch).float())
        
        
        i += 1
        print(str(i) + "/" + str(batches_per_epoch) + " | " + str(Accuracy))
        
 
# evaluate trained model with test set
#with torch.no_grad():
 #   break
    #y_pred = model(X)
#accuracy = (y_pred.round() == y).float().mean()
#print("Accuracy {:.2f}".format(accuracy * 100))






#%%%


def train_one_epoch(epoch_index, tb_writer):
   # global inputs
    #global labels
    
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    correct = 0
    amount = 0
    for i, data in enumerate(train):
        
        #print(i)
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        

        # Make predictions for this batch
        outputs = model(inputs.long())
        labels = labels.view(-1, 1)
        
        correct += (outputs.argmax(1) == labels).float().sum()
        amount = amount + len(outputs)
        
        # Compute the loss and its gradients
        loss = loss_function(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            accuracy = correct / amount
            correct = 0
            amount = 0
            
            
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {} Accuracy: {}'.format(i + 1, last_loss, accuracy))
            tb_x = epoch_index * len(train) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    model.to(device)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

   # running_vloss = 0.0
   # for i, vdata in enumerate(test):
   #     vinputs, vlabels = vdata
   ##     voutputs = model(vinputs)
   #     vloss = loss_function(voutputs, vlabels)
   #     running_vloss += vloss
#
   # avg_vloss = running_vloss / (i + 1)
    print('LOSS train {}'.format(avg_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state

    epoch_number += 1

#%%%



from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


#%%%

maxlen = 50
embedding_dim = 64
hidden_dim = 64

class LSTM_Network(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        
        print("LSTM Layer Vocab size: " + str(vocab_size))
        
        super(LSTM_Network, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to label space
        self.hidden2tag = nn.Linear(hidden_dim, 1)
    
    def forward(self, sentence):
        
       # print(sentence)
       
       
       
        embeds = self.word_embeddings(sentence)
        #embeds = torch.transpose(embeds,0,1)
        
        
        #print(embeds.reshape(-1,embedding_dim).size())
        
        embeds = embeds.view(len(sentence), -1, embedding_dim)
        
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :] 

        
        #print(lstm_out.size())
        
        
        
        
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space)
        return tag_scores
        
#model = LSTM_Network(embedding_dim, hidden_dim, d_train.tokenizer.vocab_size, 2)
loss_function = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr = 0.01)



def train_one_epoch(epoch_index, tb_writer):
   # global inputs
    #global labels
    
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    correct = 0
    amount = 0
    for i, data in enumerate(train):
        
        #print(i)
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        

        # Make predictions for this batch
        outputs = model(inputs.long())
        labels = labels.view(-1, 1)
        
        correct += (outputs.argmax(1) == labels).float().sum()
        amount = amount + len(outputs)
        
        # Compute the loss and its gradients
        loss = loss_function(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            accuracy = correct / amount
            correct = 0
            amount = 0
            
            
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {} Accuracy: {}'.format(i + 1, last_loss, accuracy))
            tb_x = epoch_index * len(train) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    model.to(device)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

   # running_vloss = 0.0
   # for i, vdata in enumerate(test):
   #     vinputs, vlabels = vdata
   ##     voutputs = model(vinputs)
   #     vloss = loss_function(voutputs, vlabels)
   #     running_vloss += vloss
#
   # avg_vloss = running_vloss / (i + 1)
    print('LOSS train {}'.format(avg_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state

    epoch_number += 1








#%%
batch_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        

        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
