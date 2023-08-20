
import numpy as np
# Regra da raiz quadrada
q2 = np.sqrt(p * m)
q2 = int(q2)
print('Regra da raiz quadrada: ', q2)

import numpy as np
import requests, gzip, os, hashlib
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
%pylab inline

(X, Y), (X_test, y_test) = mnist.load_data()

#Validation split
rand=np.arange(60000)
np.random.shuffle(rand)
train_no=rand[:50000]

val_no=np.setdiff1d(rand,train_no)

X_train,X_val=X[train_no,:,:],X[val_no,:,:]
y_train,y_val=Y[train_no],Y[val_no]

X_train = X_train.reshape((-1, 28*28))
X_val = X_val.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

def init(x, y):
    layer = np.random.uniform(-1, 1., size=(x,y)) / np.sqrt(x*y)
    return layer.astype(np.float32)

# sigmoid function
def sigmoid(x):
    return 1 / (np.exp(-x)+1)

# derivative of sigmoid
def d_sigmoid(x):
    return 1 / (np.exp(-x)+1) * (1 - (1 / (np.exp(-x)+1)))

# sofmax function
def softmax(x):
    exp_element = np.exp(x-x.max())
    return exp_element / np.sum(exp_element, axis=0)

# derivative of softmax
def d_softmax(x):
    exp_element = np.exp(x-x.max())
    return exp_element / np.sum(exp_element, axis=0) * (1-exp_element / np.sum(exp_element, axis=0))

# foward and backward pass
def forward_backward_pass(x, y):
    targets = np.zeros((len(y), 10), np.float32)
    targets[range(targets.shape[0]), y] = 1

    x_l1 = x.dot(l1)
    x_sigmoid = sigmoid(x_l1)

    x_l2 = x_sigmoid.dot(l2)
    out = softmax(x_l2)


    error = 2 * (out - targets) / out.shape[0] * d_softmax(x_l2)
    update_l2 = x_sigmoid.T @ error


    error = ((l2).dot(error.T)).T * d_sigmoid(x_l1)
    update_l1 = x.T @ error

    return out, update_l1, update_l2

# -------------------------- PCA -------------------------- #
# --------------------------------------------------------- #
# Calcular a matriz de covariância
cov_matrix = np.cov(X_train.T)

# Calcular os autovetores e autovalores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ordenar os autovetores em ordem decrescente dos autovalores
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Escolher o número de componentes principais
num_components = 155

# Selecionar as componentes principais
principal_components = sorted_eigenvectors[:, :num_components]

# Projetar os dados nas componentes principais
X_train_pca = np.dot(X_train, principal_components)

# Projetar os dados nas componentes principais
X_val_pca = np.dot(X_val, principal_components)

# Projetar os dados nas componentes principais
X_test_pca = np.dot(X_test, principal_components)

# -------------------------- PCA -------------------------- #
# --------------------------------------------------------- #

epochs = 10000
lr = 0.001
batch = 128

np.random.seed(42)
l1 = init(num_components, q2)
l2 = init(q2, 10)

accuracies, losses, val_accuracies, val_losses, test_accuracies, test_losses = [], [], [], [], [], []

for i in range(epochs):
    sample = np.random.randint(0, X_train_pca.shape[0], size=(batch))
    x = X_train_pca[sample]
    y = y_train[sample]

    out, update_l1, update_l2 = forward_backward_pass(x, y)

    category = np.argmax(out, axis=1)
    accuracy = (category == y).mean()
    accuracies.append(accuracy)

    loss = ((category - y)**2).mean()
    losses.append(loss.item())

    l1 = l1 - lr*update_l1
    l2 = l2 - lr*update_l2

    if(i%20 == 0):
        val_out = np.argmax(softmax(sigmoid(X_val_pca.dot(l1)).dot(l2)), axis=1)
        val_acc = (val_out == y_val).mean()
        val_accuracies.append(val_acc.item())
        val_loss = ((val_out - y_val)**2).mean()
        val_losses.append(val_loss.item())
    if(i%500 == 0): print(f'For {i}th epoch: train accuracy: {accuracy:.3f} | validation accuracy:{val_acc:.3f}')



test_out=np.argmax(softmax(sigmoid(X_test_pca.dot(l1)).dot(l2)),axis=1)
test_acc=(test_out==y_test).mean().item()
print(f'Test accuracy = {test_acc*100:.2f}%')
