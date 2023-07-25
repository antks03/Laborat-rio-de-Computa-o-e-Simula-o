import random
import math
import numpy as np
import scipy.stats as sp
import scipy 
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import beta
import statistics 
from scipy.stats import weibull_min, gamma, beta, qmc
import time

print("-Antonio Gabriel Freitas da Silva - 13687290-")
random.seed(12643601)
np.random.seed(12643601)
# Defina a função f(x)
def f(x):
    return np.exp(-0.32798830*x)*np.cos(0.04715069237*x)

def phi(x):
    return -0.3*x + 1

def crude_monte_carlo(f, num_amostras):
    tempo_total = 0
    num_execucoes = 10
    
    for i in range(num_execucoes):
        t0 = time.time()
        rng = qmc.Halton(d=1, scramble=False)
        x = rng.random(n=num_amostras)
        fx = f(x)
        integral = np.sum(fx) / num_amostras
        t1 = time.time()
        tempo = t1 - t0
        tempo_total += tempo
        var = np.sum((fx - integral)**2) / num_amostras 
    print("CRUDE:")
    print(f'Área: {integral:.6f} \nVariância: {var:.6f}')
    
    tempo_medio = tempo_total / num_execucoes
    print(f'Tempo médio de execução: {tempo_medio:.6f}s \n')


def hit_or_miss(num_amostras):
    tempo_total = 0
    num_execucoes = 10
    for i in range(num_execucoes):
        t0 = time.time()
        sampler= qmc.Halton(d=2, scramble=False)
        sample = sampler.random(n=num_amostras)
        a, b = 0, 1  # limites do intervalo em que a função está definida
        S = (b - a)**2  # área total do retângulo
        num_hits = 0
        for i in range(num_amostras):
            x = sample[i][0]
            y = sample[i][1]
            if y <= f(x):
                num_hits += 1
        integral = S * num_hits / num_amostras
        t1 = time.time()
        tempo = t1-t0
        tempo_total += tempo
        var = (integral)*(1-integral)
    print("HIT OR MISS:")
    print(f'Área: {integral:.6f} \nVariância: {var:.6f}')
    tempo_medio = tempo_total / num_execucoes
    print(f'Tempo médio de execução: {tempo_medio:.6f}s \n')

def importance_sampling(f, a, b, num_amostras):
    tempo_total = 0
    num_execucoes = 10
    for i in range(num_execucoes):
        # amostras da distribuição beta
        t0 = time.time()
        k, theta = 1, 1.1 # parametros da distribuição
        g = sp.beta(k, theta)
        sampler_x = qmc.Halton(d=1, scramble=False)
        # calcular a estimativa inicial da integral com num_amostras
        x = sampler_x.random(n=num_amostras)
        fx = f(x)
        gx = g.pdf(x)
        integral = np.sum(fx/gx)/num_amostras 
        t1 = time.time()
        tempo = t1 - t0
        tempo_total += tempo
    # calcular a nova amostragem para um erro relativo menor do que 0.05%
    variance = np.sum(gx*(fx/gx - integral)**2)/num_amostras
    # imprimir o resultado
    print("IMPORTANCE SAMPLING:")
    print(f"Área: {integral:.6f} \nVariância: {variance:.6f}")
    tempo_medio = tempo_total / num_execucoes
    print(f'Tempo médio de execução: {tempo_medio:.6f}s \n')

def control_variates(f, g, a, b, num_amostras):
    tempo_total = 0
    num_execucoes = 10
    # integrando g(x) no intervalo [a, b]^2
    for i in range(num_execucoes):
        t0 = time.time()
        integral_g = 0.85
        soma = 0
        fx = []
        gx = []
        sampler_x = qmc.Halton(d=2, scramble=False)
        for i in range(num_amostras):
            amostras = sampler_x.random(n=1)[0]
            fx.append(f(amostras))
            gx.append(g(amostras))
            soma += fx[i] - gx[i] + integral_g
        area = np.mean(soma)/num_amostras
        t1 = time.time()
        tempo = t1 - t0
        tempo_total += tempo
        # estimando a variância do estimador
        var_f = np.var(fx)
        var_g = np.var(gx)
        cov_fg = np.cov(fx, gx)[0][1]
        variancia_estimador = abs(var_f - 2 * cov_fg + var_g)
    
    print("CONTROL VARIATES:")
    print(f'Área: {area} \nVariância: {variancia_estimador:.6f}')
    tempo_medio = tempo_total / num_execucoes
    print(f'Tempo médio de execução: {tempo_medio:.6f}s \n')
    
def Main():
    a, b = 0, 1
    num_amostras = 10000
    crude_monte_carlo(f, num_amostras)
    hit_or_miss(num_amostras)
    importance_sampling(f, a, b, num_amostras)
    control_variates(f, phi, a, b, num_amostras)

Main()

