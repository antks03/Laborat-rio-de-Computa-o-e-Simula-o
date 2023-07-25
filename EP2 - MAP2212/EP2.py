import random
import math
import numpy as np
import scipy.stats as sp
import scipy 
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import beta
import statistics 
from scipy.stats import weibull_min, gamma, beta

def calculo_n(var, erro):   
    n = var * (1.96/erro)**2
    print(f'Nova amostragem: {int(n)}\n')

# Defina a função f(x)
def f(x):
    return np.exp(-0.32798830*x)*np.cos(0.04715069237*x)

def phi(x):
    return -0.3*x + 1

def crude_monte_carlo(f, a, b, num_amostras):
    x = np.random.uniform(a, b, num_amostras)
    fx = f(x)
    integral = np.sum(fx) / num_amostras
    var = np.sum((fx - integral)**2) / num_amostras
    erro = 0.0005*integral
    print("CRUDE:")
    print(f'Área: {integral:.6f} \nVariância: {var:.6f}')
    calculo_n(var, erro)   

def hit_or_miss(num_amostras):
    a, b = 0, 1  # limites do intervalo em que a função está definida
    S = (b - a)**2  # área total do retângulo
    num_hits = 0
    for i in range(num_amostras):
        x = random.uniform(a, b)
        y = random.uniform(a, b)
        if y <= f(x):
            num_hits += 1
    integral = S * num_hits / num_amostras
    var = (integral)*(1-integral)
    erro = 0.0005*integral
    print("HIT OR MISS:")
    print(f'Área: {integral:.6f} \nVariância: {var:.6f}')
    calculo_n(var, erro)

def relative_error(nova_integral, antiga_integral):
    return np.abs((nova_integral - antiga_integral)/antiga_integral)

def importance_sampling(f, a, b, num_amostras):
    # amostras da distribuição beta
    k, theta = 1, 1.1 # parametros da distribuição
    g = sp.beta(k, theta)

    # calcular a estimativa inicial da integral com num_amostras
    x = g.rvs(num_amostras)
    fx = f(x)
    gx = g.pdf(x)
    integral = np.sum(fx/gx)/num_amostras 

    # calcular a nova amostragem para um erro relativo menor do que 0.05%
    variance = np.sum(gx*(fx/gx - integral)**2)/num_amostras
    # loop para ajustar a estimativa com a amostragem
    inferior_limite = num_amostras
    superior_limite = num_amostras*10
    while True:
        ponto_medio = (inferior_limite + superior_limite)//2
        x = g.rvs(ponto_medio)
        fx = f(x)
        gx = g.pdf(x)
        nova_integral = np.sum(fx/gx)/ponto_medio
        rel_error = relative_error(nova_integral, integral)

        if rel_error < 0.0005:
            break
        elif nova_integral > integral:
            superior_limite = ponto_medio
        else:
            inferior_limite = ponto_medio
        
    # imprimir o resultado
    print("IMPORTANCE SAMPLING:")
    print(f"Área: {integral:.6f} \nVariância: {variance:.6f} \nNova amostragem: {ponto_medio} \n")

def control_variates(f, g, a, b, num_amostras):
    # integrando g(x) no intervalo [a, b]
    integral_g = 0.85
    soma = 0
    fx = []
    gx = []
    for i in range(num_amostras):
        amostras = random.uniform(a, b)
        fx.append(f(amostras))
        gx.append(g(amostras))
        soma += fx[i] - gx[i] + integral_g
    area = soma/num_amostras
    # estimando a variância do estimador
    var_f = np.var(fx)
    var_g = np.var(gx)
    cov_fg = np.cov(fx, gx)[0][1]
    variancia_estimador = abs(var_f - 2 * cov_fg + var_g)
    erro = 0.0005*area
    
    print("CONTROL VARIATES:")
    print(f'Área: {area:.6f} \nVariância: {variancia_estimador:.6f}')
    calculo_n(variancia_estimador, erro)

def Main():
    a, b = 0, 1
    num_amostras = 10000
    crude_monte_carlo(f, a, b, num_amostras)
    hit_or_miss(num_amostras)
    importance_sampling(f, a, b, num_amostras)
    control_variates(f, phi, a, b, num_amostras)

np.random.seed(9)
random.seed(9)
Main()