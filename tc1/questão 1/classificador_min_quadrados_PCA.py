import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

def least_squares_classification(X, y, alpha):
    # Adiciona uma coluna de 1s para representar o termo de viés
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Calcula os coeficientes usando a fórmula dos mínimos quadrados com regularização de Ridge
    theta = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

    return theta

# Carregando o conjunto de dados MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Ajusta a forma dos dados
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalizar os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

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
num_components = 200

# Selecionar as componentes principais
principal_components = sorted_eigenvectors[:, :num_components]

# Projetar os dados nas componentes principais
X_train_pca = np.dot(X_train, principal_components)

# Projetar os dados nas componentes principais
X_test_pca = np.dot(X_test, principal_components)

# -------------------------- PCA -------------------------- #
# --------------------------------------------------------- #

# Converte os rótulos para uma representação one-hot
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]

# Define o valor de regularização (alpha)
alpha = 0.01

# Realiza a classificação usando mínimos quadrados com regularização de Ridge
theta = least_squares_classification(X_train_pca, y_train_onehot, alpha)

# Adiciona uma coluna de 1s aos dados de teste
X_test = np.concatenate((np.ones((X_test_pca.shape[0], 1)), X_test_pca), axis=1)

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
