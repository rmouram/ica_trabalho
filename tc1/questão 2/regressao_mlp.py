import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# Read xlsx file
ws = pd.read_excel("Real_estate_valuation_dataset.xlsx", engine='openpyxl')
# Drop useless column
ws = ws.drop('No', axis=1)
# Convert to numpy ndarray
data = np.array(ws)

X = data[:, :-1]
Y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def init(x, y):
    layer = np.random.uniform(-1, 1., size=(x, y)) / np.sqrt(x * y)
    return layer.astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def forward_backward_pass(x, y):
    x_l1 = x.dot(l1)
    x_sigmoid = sigmoid(x_l1)

    x_l2 = x_sigmoid.dot(l2)
    out = x_l2
    
    error = 2 * (out - y) / out.shape[0]
    update_l2 = x_sigmoid.T @ error
    
    error = (error @ l2.T) * d_sigmoid(x_sigmoid)
    update_l1 = x.T @ error
    
    return out, update_l1, update_l2

epochs = 20000
lr = 0.001
batch = 32
num_hidden_neurons = 20

np.random.seed(42)
l1 = init(X_train.shape[1], num_hidden_neurons)
l2 = init(num_hidden_neurons, 1)

losses = []

#y = y.reshape((-1, 1))

for i in range(epochs):
    sample = np.random.randint(0, X_train.shape[0], size=(batch))
    x = X_train[sample]
    y = y_train[sample].reshape((-1, 1))

    out, update_l1, update_l2 = forward_backward_pass(x, y)

    loss = mean_squared_error(y, out)  # Calculate MSE
    losses.append(loss)

    l1 -= lr * update_l1
    l2 -= lr * update_l2

    if i % 100 == 0:
        print(f'Epoch {i}, Loss: {loss}')


X_test = X_test.reshape((-1, X_train.shape[1]))
y_test = y_test.reshape((-1, 1))

# Teste
def test(x, y, l1, l2):
    x_l1 = x.dot(l1)
    x_sigmoid = sigmoid(x_l1)

    x_l2 = x_sigmoid.dot(l2)
    out = x_l2
    
    return out

# Calculando as previsões para o conjunto de testes
y_pred_test = test(X_test, y_test, l1, l2)

# Calculando o MSE para o conjunto de testes
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)

# Calculando o MAE para o conjunto de testes
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Test MAE:", mae_test)

r2 = r2_score(y_test, y_pred_test)
print("R2:", r2)

rmse = np.sqrt(mse_test)
print("Raiz do erro médio quadrado (RMSE):", rmse)
