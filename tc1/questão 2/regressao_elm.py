import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ELMRegressor:
    def __init__(self, num_hidden_neurons):
        self.num_hidden_neurons = num_hidden_neurons
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # Inicialização aleatória dos pesos e vieses
        self.weights_input_hidden = np.random.rand(num_features, self.num_hidden_neurons)
        self.bias_hidden = np.random.rand(1, self.num_hidden_neurons)

        
        # Calcula as saídas da camada oculta
        hidden_output = self._relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)

        # Calcula os pesos da camada de saída usando a pseudo-inversa
        self.weights_hidden_output = np.dot(np.linalg.pinv(hidden_output), y)

    def predict(self, X):
        hidden_output = self._relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        predictions = np.dot(hidden_output, self.weights_hidden_output)
        return predictions

# Read xlsx file
ws = pd.read_excel("Real_estate_valuation_dataset.xlsx", engine='openpyxl')
# Drop useless column
ws = ws.drop('No', axis=1)
# Convert in numpy ndarray
data = np.array(ws)

X = data[:, :-1]
Y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Crie uma instância do regressor ELM e ajuste-o aos dados de treinamento
elm_regressor = ELMRegressor(num_hidden_neurons=16)
elm_regressor.fit(X_train, y_train)

# Faça previsões nos dados de teste
predictions = elm_regressor.predict(X_test)
#print("predictions: ", predictions)

# Calcule o Mean Squared Error (MSE)
mse = np.mean((y_test - predictions)**2)
print("Mean Squared Error:", mse)

# Calculando o MAE para o conjunto de testes
mae_test = mean_absolute_error(y_test, predictions)
print("Test MAE:", mae_test)
r2 = r2_score(y_test, predictions)
print("R2:", r2)
rmse = np.sqrt(mse)
print("Raiz do erro médio quadrado (RMSE):", rmse)
