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

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Parameters
Ng = 500 # Number of iterations
M = 10
limites = [-6, 20]
k = 5
y = pot
data_values = []
coeficientes = []

for m in range(M):
    # Define the variables
    
    
    x_best = np.random.uniform(limites[0], limites[1], size=(k+1,))    
    #B = np.polyfit(v, y, k)
    ypred = np.polyval(x_best, v)
          
    erro = y - ypred
    SEQ = np.sum(erro ** 2)

    Fbest = SEQ
    
    cands = []
    aptidao = []
    for t in range(1, Ng + 1):
        
        x_cand = np.random.uniform(limites[0], limites[1], size=(k+1,))
        cands.append(x_cand)
        #B = polyfit(v, y, k)
        ypred = np.polyval(x_cand, v)
        erro = y - ypred
        
        SEQ = np.sum(erro ** 2)
    
        Fcand = SEQ
        
        if Fcand < Fbest:
            x_best = x_cand
            Fbest = Fcand
    
        aptidao.append(Fbest)
    
    ypred = np.polyval(x_best, v)
    erro = y - ypred
    R2 = r2(y, erro)
    R2_ajustado = r2_ajustado(y, erro, len(y), k + 1)
    AIC = aic(v, k, erro)
    data_values.append((R2, R2_ajustado, AIC, m))
    coeficientes.append((x_best, m))
    
    print("Best solution:", x_best)
    print("Best fitness:", Fbest)
    
    plt.figure()
    plt.plot(aptidao, linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('y=f(x)')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('M ={}'.format(m))
    
    ypred = np.polyval(x_best, v)
    plt.figure()
    plt.plot(v, ypred, 'ro')
    plt.xlabel('v')
    plt.ylabel('Potência')
    plt.title('M ={}'.format(m))
    
    plt.show()

