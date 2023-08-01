import scipy.special as stats
import numpy as np
import re
import scipy.optimize as optimize
from scipy.stats import chi2

#função responsável por receber um v qualquer e devolver a cumulada desse v
def Cal_v(pontos,v):
    #se v está abaixo do menor ponto então a sua cumulada é 0
    if v<pontos[0]:
        return 0
    #se v está acima do maior ponto então a sua cumulada é 1
    if v>=pontos[-1]:
        return 1
    #se v está entre o menor e o maior ponto então colocamos v no vetor de pontos,achamos a sua posição e calculamos a cumulada
    pontos2=pontos
    pontos2.append(v)
    pontos2.sort()
    x=pontos2.index(v)
    return (x-1)/len(pontos)


#calcula o valor de 1/beta(A) por meio de uma gamma
def const_norm(A):
    soma = 0
    prod_gamma = 1
    for k in range(3):
        soma += (A[k])
        #calcula o valor da gamma de A[k] e multiplica com os outros valores
        prod_gamma *= stats.gamma(A[k])
    #calcula a gamma da soma dos valores de A
    sum_gamma = stats.gamma(soma)
    return sum_gamma/prod_gamma

#Calcular o A
def Cal_A(x,y):
    z=[]
    for k in range(len(x)):
        #soma cada x[i] e y[i]
        A=int(x[k])+int(y[k])
        z.append(A)
    return z

def potencial(theta,A,gamma):
    prod=1
    #calcula (theta)1 elevado a (A)1 -1 e multiplica com o valor calculado para theta2 e theta3
    for k in range(3):
        prod*=theta[k]**(int(A[k])-1)
    #multiplica o valor gerado na multiplicação por gamma
    prod=prod*gamma
    return prod

#gera os valores de t(theta|A)
def Cal_T(n,A):
    #calcula o valor da gamma de A
    pontos=[]
    potencial_x=0
    gamma=const_norm(A)
    #gera n thetas(cada theta possui um theta1,theta2,theta3) utilizando o vetor A
    x=np.random.dirichlet(A,n)
    for z in range(n):
        potencial_x=potencial(x[z],A,gamma)
        #adiciona ao vetor de pontos
        pontos.append(potencial_x)
    pontos.sort()
    return pontos

#calcula número de pontos necessários
def Cal_n():
    #k obtido por 1/erro
    k=2000
    #variancia máxima da binomial
    var=0.25
    #calcula o n necessário e transforma em int
    n=((1.95)**2)*var/0.0005**2
    n=int(n)
    return n

def theta_star(A):

    def Ho(theta1):
        theta2=1-theta1-(1-(np.sqrt(theta1)))**2
        theta3=(1-(np.sqrt(theta1)))**2
        return [theta1,theta2,theta3]

    def s_star(x):
        return (-potencial(Ho(x),A,const_norm(A)))

    maximiza=optimize.minimize_scalar(s_star,bounds=(0.0, 1.0), method='Bounded')
    return -maximiza.fun

def sev_calc(e_valor):
    graus_de_lib=1
    tamanho_espaco=2
    ev=1-e_valor
    chi=chi2.cdf(chi2.ppf(ev,tamanho_espaco),graus_de_lib)
    return 1-chi

def main():
    numeros_x=[[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                    [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14],
                    [1, 15], [1, 16], [1, 17], [1, 18], [5, 0], [5, 1],
                    [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
                    [5, 9], [5, 10], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4],
                    [9, 5], [9, 6], [9, 7]]
    numeros_y=[[0, 0, 0], [1, 1, 1]]
    for i in range(len(numeros_x)):
        numeros_x[i].append(numeros_x[i][1])
        numeros_x[i][1]=20-numeros_x[i][0]-numeros_x[i][2]
    #pede valor da seed
    while True:
        try:
            seed=int(input("Digite a seed: "))
        except:
            print("Valor inválido")
        else:
            break
    np.random.seed(seed)
    #calcula o número de partições(k) e o número de pontos necessários(n)
    n=Cal_n()
    pontos=[]
    maximo=0
    e_valor=0
    Sev=0
    aceitou=0
    rejeitou=0
    erro=0
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    for j in range(2):
        for i in range(len(numeros_x)):
            A=Cal_A(numeros_x[i],numeros_y[j])
            if A[2]==0:
                print(f"x[1] = {numeros_x[i][0]}\tx[3] = {numeros_x[i][2]}\ty = {numeros_y[j]}\nDecisão do teste = anomalia(x[2]+y[2]=0)\tθ*  = NA\ne_valor = NA\tsev = NA")
                erro+=1
                continue
            maximo=theta_star(A)
            pontos=Cal_T(n,A)
            e_valor=Cal_v(pontos,maximo)
            Sev=sev_calc(e_valor)
            if Sev<0.05:
              rejeita='Rejeita Ho'
              aceitou+=1
            else:
              rejeita='Aceita Ho'
              rejeitou+=1
            print(f"x[1] = {numeros_x[i][0]}\tx[3] = {numeros_x[i][2]}\ty = {numeros_y[j]}\nDecisão do teste = {rejeita}\tθ*  = {maximo}\ne_valor = {e_valor}\tsev = {Sev}\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            pontos=[]
        print(f"Testes rejeitados = {rejeitou} \tTestes aceitados = {aceitou}\tAnomalias = {erro}")
        erro,aceitou,rejeitou=0,0,0

main()