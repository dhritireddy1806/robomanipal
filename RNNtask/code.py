import numpy as np
import pandas as pd

def tanh(z):
    return (np.exp(z)-np.exp(-z))/((np.exp(z)+np.exp(-z)))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def clip(g, limit=5):
    return np.clip(g, -limit, limit)

vocab=100
embedd_dim=50
hidden_dim=32
outp_dim=1

class RNN():
    def __init__(self,vocab,embedd_dim,hidden_dim,outp_dim):
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.embedd_dim = embedd_dim
        self.outp_dim = outp_dim
        self.E=np.random.randn(vocab, embedd_dim)*0.01
        self.Wxh=np.random.randn(hidden_dim,embedd_dim)*0.01
        self.Whh=np.random.randn(hidden_dim,hidden_dim)*0.01
        self.Why=np.random.randn(1,hidden_dim,)*0.01
        self.bh=np.zeros((hidden_dim,1))
        self.by=np.zeros((1,1))

    def forwardprop(self,seq):
        self.hs=[]
        self.xs=[]
        h_prev=np.zeros((self.hidden_dim,1))
        for word in seq:
            xt=self.E[word].reshape(-1,1)
            ht=tanh(self.Wxh@xt+self.Whh@h_prev+self.bh)
            self.hs.append(ht)
            self.xs.append(xt)
            h_prev=ht
        h_final=self.hs[-1]
        z=self.Why@h_final+self.by
        yhat=sigmoid(z)

        return yhat

    def backprop(self,yhat,y,seq):
        T=len(self.hs)
        dWxh=np.zeros_like(self.Wxh)
        dWhh=np.zeros_like(self.Whh)
        dWhy=np.zeros_like(self.Why)
        dbh=np.zeros_like(self.bh)
        dby=np.zeros_like(self.by)
        dE = np.zeros_like(self.E)

        dz=yhat-y
        h_final=self.hs[-1]
        dWhy=dz @ h_final.T
        dby=dz
        dh=self.Why.T @ dz

        for i in reversed(range(T)):
            ht=self.hs[i]
            if i==0:
                h_prev=np.zeros((self.hidden_dim,1))
            else:
                h_prev=self.hs[i-1]
            xt=self.xs[i]
            dtanh=(1-ht**2)*dh
            dWxh+=dtanh @ xt.T
            dWhh+=dtanh @ h_prev.T
            dbh+=dtanh
            dh=self.Whh.T @ dtanh
            word = seq[i]
            dE[word]+= (self.Wxh.T @ dtanh).flatten()
        return dWxh,dWhh,dWhy,dbh,dby,dE
    
    def update(self,dWxh,dWhh,dWhy,dbh,dby,dE,lr):
        dWxh = clip(dWxh)
        dWhh = clip(dWhh)
        dWhy = clip(dWhy)
        dbh = clip(dbh)
        dby = clip(dby)
        dE = clip(dE)
        self.Wxh-=lr*dWxh
        self.Whh-=lr*dWhh
        self.Why-=lr*dWhy
        self.bh-=lr*dbh
        self.by-=lr*dby
        self.E-= lr * dE
    
epochs=10
    
df=pd.read_csv("Sentiment Analysis Dataset.csv", encoding="latin1")

for col in df.columns:
    df[col]=df[col].fillna(df[col].mode()[0])

X=df['SentimentText'].str.lower().str.strip()
y=df["Sentiment"].astype(int).values

import re
X=X.apply(lambda text:re.sub(r'[^a-zA-Z\s]', '', text))

X=X.apply(lambda text:text.split())
vocab={"<PAD>": 0}
index=1
for sent in X:
    for word in sent:
        if word not in vocab:
            vocab[word]=index
            index+=1
vocab_size=len(vocab)+1
rnn=RNN(vocab_size,embedd_dim,hidden_dim,outp_dim)

encoded_sent=[]
for sent in X:
    encoded=[]
    for word in sent:
        encoded.append(vocab[word])
    encoded_sent.append(encoded)

max_len=max(len(s) for s in encoded_sent)

padded=[]
for sent in encoded_sent:
    while len(sent)<max_len:
        sent.append(0)
    padded.append(sent)

X=np.array(padded)
y=np.array(y)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
for n in range(epochs):
    epoch_loss=0
    correct=0
    for sent, label in zip(X_train,y_train):
        yhat=rnn.forwardprop(sent)

        dWxh,dWhh,dWhy,dbh,dby,dE =rnn.backprop(yhat,label,sent)
        loss = -(label*np.log(yhat+1e-8) +
         (1-label)*np.log(1-yhat+1e-8))

        epoch_loss += loss.item()
        rnn.update(dWxh,dWhh,dWhy,dbh,dby,dE,lr=0.001)
        
        prediction = 1 if yhat.item() >= 0.5 else 0
        if prediction ==int(label):
            correct+=1
    epoch_loss/=len(X_train)
    accuracy = correct / len(X_train)
    print(f"Loss of {n} epoch:",epoch_loss)
    print(f"Accuracy of {n} epoch:",accuracy*100)

correct = 0

for sent, label in zip(X_test, y_test):

    yhat = rnn.forwardprop(sent)

    prediction = 1 if yhat >= 0.5 else 0

    if prediction == label:
        correct += 1

print("Test Accuracy:", correct/len(X_test))