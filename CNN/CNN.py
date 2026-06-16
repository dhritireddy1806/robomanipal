import numpy as np
import gzip

def load_images(filename):
    with gzip.open(filename,'rb') as f:
        data=np.frombuffer(f.read(),np.uint8, offset=16)
        return data.reshape(-1,28,28)
    
def load_label(filename):
    with gzip.open(filename,'rb') as f:
        return np.frombuffer(f.read(),np.uint8,offset=8)
    
def one_hot(file):
    arr=np.zeros((10,1))
    arr[file]=1
    return arr

X_train=load_images('train-images-idx3-ubyte.gz')
y_train=load_label('train-labels-idx1-ubyte.gz')
X_test=load_images('t10k-images-idx3-ubyte.gz')
y_test=load_label('t10k-labels-idx1-ubyte.gz')

X_train=X_train/255.0
X_test=X_test/255.0

X_train=[x.reshape(28,28) for x in X_train]
X_test=[x.reshape(28,28) for x in X_test]
print ("Shape of X_train:",X_train[0].shape)

filters=np.random.randn(8,3,3)*0.01

X_train = X_train[:1000]
y_train = y_train[:1000]
X_test = X_test[:1000]
y_test = y_test[:1000]

img=X_train[0]
label=y_train[0]

def convo(img,filters):
    fm=np.zeros((26,26,8))
    for k in range(8):
        fil=filters[k]
        for i in range(26):
            for j in range(26):
                patch=img[i:i+3,j:j+3]
                fm[i,j,k]=np.sum(patch*fil)
    return fm

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    return (x > 0).astype(int)

def softmax(z):
    exp_z=np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z)


def maxpooling(fm):
    pooled=np.zeros((13,13,8))
    for k in range(8):
       for i in range(13):
           for j in range(13):
                region=fm[i*2:i*2+2,j*2:j*2+2,k]
                pooled[i,j,k]=np.max(region)
    return pooled

def maxpool_backprop(delta_pool, relu_out):

    delta_relu = np.zeros_like(relu_out)

    for k in range(8):

        for i in range(13):

            for j in range(13):

                region = relu_out[
                    i*2:i*2+2,
                    j*2:j*2+2,
                    k
                ]

                max_pos = np.unravel_index(
                    np.argmax(region),
                    region.shape
                )

                delta_relu[
                    i*2 + max_pos[0],
                    j*2 + max_pos[1],
                    k
                ] = delta_pool[i,j,k]

    return delta_relu


def conv_backprop(delta_conv, img):

    dfilters = np.zeros((8,3,3))

    for k in range(8):

        for i in range(26):

            for j in range(26):

                patch = img[i:i+3,j:j+3]

                dfilters[k] += (
                    patch *
                    delta_conv[i,j,k]
                )

    return dfilters

w1 = np.random.randn(128,1352)*0.01
b1=np.zeros((128,1))*0.01
w2 = np.random.randn(10,128)*0.01
b2=np.zeros((10,1))*0.01

eta=0.01
epochs=10

for e in range(epochs):
    correct=0
    for img,label in zip(X_train,y_train):
        # CNN feature extraction

        cout = convo(img, filters)
        rout = relu(cout)
        p = maxpooling(rout)
        flat = p.flatten().reshape(-1,1)
        z1 = np.dot(w1, flat) + b1
        h = relu(z1)
        z2 = np.dot(w2, h) + b2
        output = softmax(z2)

        y = one_hot(label)
        delta2 = output - y

        dw2 = np.dot(delta2, h.T)

        db2 = delta2
        delta1 = np.dot(w2.T, delta2) * relu_prime(z1)
        

        dw1 = np.dot(delta1, flat.T)

        db1 = delta1
        # backprop into flattened layer

        delta_flat = np.dot(w1.T, delta1)

# reshape to pooled shape

        delta_pool = delta_flat.reshape(13,13,8)

# maxpool backprop

        delta_relu = maxpool_backprop(
        delta_pool,
        rout
        )

# relu backprop

        delta_conv = (
          delta_relu *
          relu_prime(cout)
          )

# convolution filter gradients

        dfilters = conv_backprop(
        delta_conv,
        img
        )
        w1-=eta*dw1
        b1-=eta*db1
        w2-=eta*dw2
        b2-=eta*db2
        filters -= eta * dfilters
        
        pred=np.argmax(output)
        if pred==label:
            correct+=1
    print("Epoch:",e+1,"\t Accuracy:",correct/len(X_train))
print("Train accuracy:",correct/len(X_train))

correct=0

for img,label in zip(X_test,y_test):

    out = convo(img, filters)
    out = relu(out)
    p = maxpooling(out)

    flat = p.flatten().reshape(-1,1)

    z1 = np.dot(w1, flat) + b1
    h = relu(z1)

    z2 = np.dot(w2, h) + b2

    output = softmax(z2)

    pred = np.argmax(output)

    if pred == label:
        correct += 1

print("Test Accuracy:", correct/len(X_test))