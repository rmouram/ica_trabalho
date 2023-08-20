import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('aerogerador.dat')

v_aero = data[:, 0]
P = data[:, 1]

# Cria o scatterplot
plt.scatter(v_aero, P, color='blue', marker='o', facecolors='none', edgecolors='blue', linewidths=0.5)

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

def pso_seq(b_aerogerador, v_aero, P):
    SEQ = []
    Ng = b_aerogerador.shape[0]
    for i in range(Ng):
        ypred = np.polyval(b_aerogerador[i, :], v_aero)
        y = P
        erro = y - ypred
        SEQ.append(np.sum(np.abs(erro)))
    F = SEQ
    return F

Ng = 150
Np = 80
c1 = 2.05
c2 = c1
w = 0.6
y = P
limites = [-6, 10]
k = 5
M = 10
data_values = []
coeficientes = []

for m in range(M):
    
    v = np.zeros((Np, 6))
    b_aerogerador = uniform.rvs(loc=limites[0], scale=limites[1] - limites[0], size=(Np, 6))
    
    b_best = b_aerogerador.copy()
    Fbest = pso_seq(b_best, v_aero, P)
    Fbest = np.array(Fbest)
    
    Fmin = np.min(Fbest)
    I = np.argmin(Fbest)
    g_best = b_best[I, :]
    
    aptidao = np.zeros(Ng)
    for t in range(Ng):
        iteracao = t
    
        Vcog = np.random.rand(Np, 6) * (b_best - b_aerogerador)
        Vsoc = np.random.rand(Np, 6) * (g_best - b_aerogerador)
    
        v = w * v + c1 * Vcog + c2 * Vsoc
        b_aerogerador = b_aerogerador + v
    
        b_aerogerador = np.clip(b_aerogerador, limites[0], limites[1])
    
        Fcand = pso_seq(b_aerogerador, v_aero, P)
        Fcand = np.array(Fcand)
        DF = np.array(Fcand) - np.array(Fbest)
        I_better = np.where(DF <= 0)[0]
        I_worse = np.where(DF > 0)[0]
    
        b_best[I_better] = b_aerogerador[I_better]
        Fbest[I_better] = Fcand[I_better]

        b_best[I_worse] = b_best[I_worse]
        Fbest[I_worse] = Fbest[I_worse]
        
        F_gbest = np.min(Fbest)
        I = np.argmin(Fbest)
        g_best = b_best[I, :]
    
        aptidao[t] = F_gbest

    ypred = np.polyval(g_best, v_aero)
    erro = y - ypred
    R2 = r2(y, erro)
    R2_ajustado = r2_ajustado(y, erro, len(y), k + 1)
    AIC = aic(v, k, erro)
    data_values.append((R2, R2_ajustado, AIC, m))
    coeficientes.append((g_best, m))
    
    print("Melhor posição:", g_best)
    print("Melhor aptidão:", F_gbest)
    
    ypred = np.polyval(g_best, v_aero)
    plt.figure()
    plt.plot(v_aero, ypred, 'ro')
    plt.xlabel('v')
    plt.ylabel('Potência')
    plt.title('M ={}'.format(m))
    
    plt.figure()
    plt.plot(aptidao, linewidth=1)
    plt.xlabel('Iteração')
    plt.ylabel('Aptidão')
    plt.title('M ={}'.format(m))
    
    plt.show()
