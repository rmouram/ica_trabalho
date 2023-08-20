import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

class LogisticPerceptron:
    def __init__(self, num_features, num_classes, learning_rate=0.01, num_epochs=100):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.zeros((num_features + 1, num_classes))  # +1 for the bias term

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        activations = np.dot(np.insert(x, 0, 1), self.weights)
        probabilities = self.sigmoid(activations)
        return np.argmax(probabilities)

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Inserting bias term
        y = np.eye(self.num_classes)[y]  # One-hot encoding
        for _ in range(self.num_epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                activations = np.dot(x, self.weights)
                probabilities = self.sigmoid(activations)
                error = target - probabilities
                delta = self.learning_rate * np.outer(x, error)
                self.weights += delta

# Carregando o conjunto de dados MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Ajusta a forma dos dados
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalizar os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Criar o objeto do perceptron logístico
num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))
perceptron = LogisticPerceptron(num_features=num_features, num_classes=num_classes)

# Treinar o perceptron
perceptron.train(X_train, y_train)

# Realizar previsões no conjunto de teste
predictions = []
for sample in X_test:
    prediction = perceptron.predict(sample)
    predictions.append(prediction)

# Calcular a precisão das previsões
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
