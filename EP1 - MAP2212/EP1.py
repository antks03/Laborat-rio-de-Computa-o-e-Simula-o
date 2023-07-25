import numpy as np
import matplotlib.pyplot as plt
import random
from math import ceil
from matplotlib.patches import Circle

def T(x):
    norma = np.linalg.norm(x)
    if norma <= 1:
        return 1
    else:
        return 0
    
def EstimativaInicial(n):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calcula o valor de T para cada ponto na malha
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = T([X[i,j], Y[i,j]])
    
    fig, ax = plt.subplots()
    circle = Circle((0,0), 1, facecolor='white', edgecolor='black', linewidth=2, alpha=0.5)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    # Amostra n pontos aleatoriamente dentro do círculo
    np.random.seed(25)
    X = np.random.uniform(-1, 1, (n, 2))
    # Cálculo da proporção
    p = sum([T(x) for x in X])/n
    pi = 4*p
    # Calcula o valor de T para cada ponto
    Y = np.array([T(x) for x in X])
    # Plota os pontos com T = 1 (dentro do círculo) em vermelho e os pontos com T = 0 (fora do círculo) em azul
    ax.scatter(X[Y==1,0], X[Y==1,1], c='r', label='Dentro do círculo')
    ax.scatter(X[Y==0,0], X[Y==0,1], c='b', label='Fora do círculo')
    # Configura a legenda e o aspecto da figura
    ax.legend()
    ax.set_aspect('equal')
    # Mostra a figura
    print("Valor estimado de pi: ", pi)
    plt.show()
    return pi

def NovaEstimativa(n):
    erro_relativo = 0.0005
    # Define o nível de confiança desejado (95%)
    z = 1.96
    # Realiza a amostragem
    pi = EstimativaInicial(n)
    # Calcula o erro relativo
    erro = (erro_relativo/4)*pi
    std = np.sqrt(pi/4 * (1-(pi/4)))
    # Verifica se o erro relativo é menor que o desejado
    n = np.ceil((z * std / erro) ** 2)
        
    # Imprime o resultado
    print(std**2)
    print("Número de amostras: ", n)
    print("Erro relativo: ", erro)

def Main():
    n = int(input("Insira as amostras desejadas: "))
    NovaEstimativa(n)

Main()