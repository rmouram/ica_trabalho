import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('aerogerador.dat')

v = data[:, 0]
pot = data[:, 1]

# Cria o scatterplot
plt.scatter(v, pot, color='blue', marker='o', facecolors='none', edgecolors='blue', linewidths=0.5)

# Adiciona rótulos e título
plt.xlabel('Velocidade do vento [m/s]')
plt.ylabel('Potência gerada [kWatts]')
plt.title('Curva de potência')

# Exibe o gráfico
plt.show()

def r2(y, erro):
    ymed = np.mean(y)
    
    SEQ = np.sum(erro ** 2)
    Syy = np.sum((y-ymed) ** 2)
    return 1 - SEQ/Syy

def r2_ajustado(y, erro, n, p):
    ymed = np.mean(y)
    
    SEQ = np.sum(erro ** 2)
    Syy = np.sum((y-ymed) ** 2)
    return 1 - (SEQ / Syy) * ((n - 1) / (n - p - 1))

def aic(v, k, erro):
    SEQ = np.sum(erro ** 2)
    return (len(v) * np.log(SEQ)) + 2*k


n = len(v)
y = pot
X = np.array([])
n = len(y)

fig, axs = plt.subplots(5, 2, figsize=(10, 20))  # Cria a grade de subplots

data_values = []
coeficientes = []

for k in range(1, 11):
    p = k + 1
    X = np.column_stack([v ** i for i in range(k + 1)])
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    coeficientes.append((B, k))
    ypred = X @ B
    erro = y - ypred
    
    R2 = r2(y, erro)
    R2_ajustado = r2_ajustado(y, erro, len(y), k + 1)
    AIC = aic(v, k, erro)
    data_values.append((R2, R2_ajustado, AIC))

    vv = np.arange(min(v), max(v) + 0.1, 0.1).reshape(-1, 1)
    XX = np.column_stack([vv ** (l - 1) for l in range(1, k + 2)])
    ypred2 = XX @ B

    row = (k - 1) // 2  # Calcula a linha do subplot na grade
    col = (k - 1) % 2   # Calcula a coluna do subplot na grade

    axs[row, col].scatter(v, y, color='blue', marker='o', label='Velocidade x Potência', facecolors='none', edgecolors='blue', linewidths=0.5)
    axs[row, col].plot(vv, ypred2, color='red', linestyle='-', label='Curva de Regressão', linewidth=0.5)

    axs[row, col].set_xlabel('Velocidade')
    axs[row, col].set_ylabel('Potência')
    axs[row, col].set_title("K = {}".format(k))
    axs[row, col].legend()

    # # Salva o gráfico como imagem
    # plt.savefig(f'grafico_{k}.png')

plt.tight_layout()  # Ajusta o layout dos subplots
plt.show()

