import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read xlsx file
ws = pd.read_excel("Real_estate_valuation_dataset.xlsx", engine='openpyxl')
# Drop useless column
ws = ws.drop('No', axis=1)
# Convert in numpy ndarray
data = np.array(ws)

X = data[:, :-1]
Y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def ridge_regression(X, y, alpha):
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


# Define o valor de regularização (alpha)
alpha = 0.01

# Realiza a regressão usando mínimos quadrados com regularização de Ridge
theta = ridge_regression(x_train, y_train, alpha)

# Adiciona uma coluna de 1s aos dados de teste
X_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)

# Realiza as predições
predictions = X_test @ theta

# Calcular o erro médio quadrado (MSE) nas previsões
mse = mean_squared_error(y_test, predictions)
print("Erro médio quadrado (MSE):", mse)
mae = mean_absolute_error(y_test, predictions)
print("Erro médio absoluto (MAE):", mae)
r2 = r2_score(y_test, predictions)
print("R2:", r2)
rmse = np.sqrt(mse)
print("Raiz do erro médio quadrado (RMSE):", rmse)
