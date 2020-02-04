import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 2
plt.imshow(train_set_x_orig[index])

train_set_x_orig[2]
train_set_x_orig[2].shape

#print (" y = " + str(train_set_y[:,index]) + classes[np.squeeze(train_set_y[:,index])].decode("utf-8"))

print train_set_y[:,2]


print train_set_x_orig.shape

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print m_train
print m_test
print num_px

#for i in range(0, 208) :

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T
train_set_x = train_set_x_flatten/255.

test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_flatten/255.

#print train_set_x_flatten[0:5,0]

#print train_set_x[0:5,0]

print test_set_x.shape
#print train_set_x
#print train_set_x.shape
#plt.imshow(train_set_x)

print train_set_x_orig[0].shape

## initialize parameters

def InitializeParameterswithZeros(dim):
    
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def propogation(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    
    cost = -(np.dot(Y, np.log(A.T)) + np.dot(np.log(1-A), (1-Y).T))/m
    
    dw = np.dot(X, (A-Y).T)/m
    db = np.
        
    grads = {"dw" : dw,
             "db" : db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    
    for i in range(num_iterations):
        
        grads, cost = propogation(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i%100 == 0:
            print(cost)
        
        grads = {"dw" : dw,
                 "db" : db}
           
    return w, b, cost, grads
    

    def predict(w, b, X):
        
    A = sigmoid(np.dot(w.T, X) + b)
        
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5: 
            A[0][i] = 0
            #print A[0][i]
        else: 
            A[0][i] = 1
            #print A[0][i]
            
    return A

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    
    w, b = InitializeParameterswithZeros(X_train.shape[0])
    w, b, cost, grads = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    print("train accuracy : " + format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy : " + format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) 
   return w, b
w, b = model(train_set_x, train_set_y, test_set_x, test_set_y, 1000, 0.005)
