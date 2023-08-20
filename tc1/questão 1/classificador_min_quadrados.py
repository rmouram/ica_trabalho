import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

def least_squares_classification(X, y, alpha):
    # Adiciona uma coluna de 1s para representar o termo de viés
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Calcula os coeficientes usando a fórmula dos mínimos quadrados com regularização de Ridge
    theta = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

    return theta

def calculate_error(X, y, theta):
    # Adiciona uma coluna de 1s para representar o termo de viés
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Realiza as predições
    predictions = X @ theta

    # Converte as predições em rótulos
    predicted_labels = np.argmax(predictions, axis=1)

    # Calcula o erro (taxa de erro)
    error = np.mean(predicted_labels != y)

    return error

# Carrega o conjunto de dados MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Ajusta a forma dos dados
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normaliza os dados dividindo por 255.0
X_train = X_train / 255.0
X_test = X_test / 255.0

# Converte os rótulos para uma representação one-hot
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]

# Define o valor de regularização (alpha)
alpha = 0.01

# Realiza a classificação usando mínimos quadrados com regularização de Ridge
theta = least_squares_classification(X_train, y_train_onehot, alpha)

# Adiciona uma coluna de 1s aos dados de teste
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)

# Realiza as predições
predictions = X_test @ theta

# Converte as predições em rótulos
predicted_labels = np.argmax(predictions, axis=1)

# # Calcula a acurácia
# accuracy = np.mean(predicted_labels == y_test) * 100
# print("Acurácia da classificação: {:.2f}%".format(accuracy))

# Calcular a precisão das previsões
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)

