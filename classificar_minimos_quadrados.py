import numpy as np
import gzip
import struct

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # Adicionar um vetor de bias aos dados
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Calculando os pesos usando a fórmula de mínimos quadrados
        X_transpose = np.transpose(X)
        self.weights = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    
    def predict(self, X):
        # Adicionar um vetor de bias aos dados
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        # Fazer a previsão usando os pesos calculados
        y_pred = X.dot(self.weights)
        return y_pred

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images / 255.0  # Normaliza as intensidades dos pixels

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Carregar os dados de treinamento e teste do MNIST
X_train = load_mnist_images('data/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')

# Achatando as imagens para vetores 1D (784 dimensões)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Criar e treinar o classificador
classifier = LinearRegression()
classifier.fit(X_train, y_train)

# Fazer previsões para os dados de teste
y_pred = classifier.predict(X_test)

# Avaliar a precisão do classificador
accuracy = np.mean(y_pred == y_test)
print("Precisão do classificador:", accuracy)

