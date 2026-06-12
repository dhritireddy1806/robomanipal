import numpy as np
import gzip

def one_hot(y):
    arr=np.zeros((10,1))
    arr[y]=1
    return arr

def load_images(filename):
    with gzip.open(filename,'rb') as f:
        data=np.frombuffer(f.read(), np.uint8,offset=16)
        return data.reshape(-1,784)# numpy figures out the dimension by doing the math

def load_label (filename):
    with gzip.open(filename, 'rb')as f:
        return np.frombuffer(f.read(),np.uint8,offset=8)

class NeuralNetwork:
    def __init__(self,sizes):
        self. layer=len(sizes)
        self.sizes=sizes
        self.biases=[np.zeros((y,1)) for y in sizes[1:]]
        self.weight=[np.random.randn(y,x)for x,y in zip(sizes[:-1], sizes[1:])]


    def forward(self, a):
        for b,w in zip(self.biases[:-1],self.weight[:-1]):
            a=relu(np.dot(w,a)+b) #hidden layers

        z=np.dot(self.weight[-1],a)+self.biases[-1]
        a=softmax(z)#o/p layer
        return a

    def SGD(self, X_train,y_train, epoch, batchsize, eta,y_test=None, X_test=None):
        n=len(X_train)
        data=list(zip(X_train,y_train))
        for i in range(epoch):
            np.random.shuffle(data)
            X_strain,y_strain= zip(*data)
            for j in range(0,n,batchsize):
                X_batch=X_strain[j:j+batchsize]
                y_batch=y_strain[j:j+batchsize]
                self.update_mini_batch(X_batch,y_batch,eta)
            if y_test is not None and X_test is not None:
                print("Epoch {0}:{1}/{2}".format(i,self.evaluate(X_test,y_test),len(y_test)))
            else:
                print("Epoch {0} complete".format(i))   

    def update_mini_batch(self,X_batch,y_batch,eta):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weight]
        m=len(X_batch)
        for x,y in zip(X_batch, y_batch):
            delta_nabla_b, delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w=[nw+dnw for nw , dnw in zip(nabla_w, delta_nabla_w)]
        self.weight=[w-(eta/m*nw) for w,nw in zip(self.weight, nabla_w)]
        self.biases=[b-(eta/m*nb)for b,nb in zip(self.biases, nabla_b)]

    def backprop(self,x,y):
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weight]
        acti=x
        activation=[x]
        zs=[]
        for b , w in zip(self.biases[:-1],self.weight[:-1]):
            z=np.dot(w,acti)+b
            zs.append(z)
            acti=relu(z)
            activation.append(acti)
        
        z=np.dot(self.weight[-1],acti)+self.biases[-1]
        zs.append(z)

        acti=softmax(z)
        activation.append(acti)
        delta = activation[-1] - y
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activation[-2].transpose())
        for i in range(2,self.layer):
            z=zs[-i]
            re=relu_prime(z)
            delta=np.dot(self.weight[-i+1].T,delta)*re
            nabla_b[-i]=delta
            nabla_w[-i]=np.dot(delta,activation[-i-1].transpose())
        return(nabla_b,nabla_w)

    def evaluate(self, X_test, y_test):
        test_result=[(np.argmax(self.forward(x,)),y) for (x,y) in zip(X_test,y_test)]
        return sum(int(x==y) for (x,y) in test_result)

def relu(z):
    return np.maximum(0,z)

def relu_prime(z):
    return (z > 0).astype(int)

def softmax(z):
    exp_z=np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z)

X_train=load_images('train-images-idx3-ubyte.gz')
y_train=load_label('train-labels-idx1-ubyte.gz')
X_test=load_images('t10k-images-idx3-ubyte.gz')
y_test=load_label('t10k-labels-idx1-ubyte.gz')

X_train=X_train/255.0
X_test=X_test/255.0

X_train = [x.reshape(784,1) for x in X_train]
X_test  = [x.reshape(784,1) for x in X_test]

net = NeuralNetwork([784,128,10])

y_train_labels = y_train.copy()
y_train=[one_hot(y) for y in y_train]
print(y_train[0].shape)
print(X_train[0].shape)
print(net.forward(X_train[0]).shape)

net.SGD(
    X_train,
    y_train,
    epoch=10,
    batchsize=32,
    eta=0.1,
    X_test=X_test,
    y_test=y_test
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = [np.argmax(net.forward(x)) for x in X_test]
y_true = y_test
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted'
)
print("Testing set")
print("Accuracy :", accuracy*100,"%")
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# using sigmoid here with relu caused the neurons to saturate at 0 or 1 causing low accuracy and the accuracy was coming around 30%
# using softmax and cross entropy helped to unsaturate the neurons and increased the accurracy, recall and precision

y_pred = [np.argmax(net.forward(x)) for x in X_train]
y_true = y_train_labels
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_true,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_true,
    y_pred,
    average='weighted'
)
print("Training set")
print("Accuracy :", accuracy*100,"%")
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)