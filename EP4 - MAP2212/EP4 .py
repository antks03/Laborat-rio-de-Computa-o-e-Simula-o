import scipy.special as stats
import numpy as np
import re
import matplotlib.pyplot as plt

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

#gera os valores de t(theta|A)
def Cal_T(n,A):
    #calcula o valor da gamma de A
    gamma=const_norm(A)
    pontos=[]
    prod=1
    #gera n thetas(cada theta possui um theta1,theta2,theta3) utilizando o vetor A
    x=np.random.dirichlet(A,n)
    for z in range(n):
        #calcula (theta)1 elevado a (A)1 -1 e multiplica com o valor calculado para theta2 e theta3 
        for k in range(3):
            prod*=x[z][k]**(int(A[k])-1)
        #multiplica o valor gerado na multiplicação por gamma
        prod=prod*gamma
        #adiciona ao vetor de pontos
        pontos.append(prod)
        #reseta a multiplicação
        prod=1
    pontos.sort()
    return pontos

#função responsável por contar quantos pontos existem abaixo dos intervalos e dividir pelo número total
def salto(pontos, kpontos):
    #gera lista para armazenar os valores obtidos
    lista_int = []
    lista_cumulada=[]
    soma=0
    #utiliza essa função para obter o número de pontos que existem no vetor pontos entre cada intervalo
    #obs:z é posto para não atrapalhar a saida da lista
    lista_int,z=np.histogram(pontos,bins=kpontos)
    #loop responsável por calcular o número de pontos que existem até certo intervalo
    for i in range(len(lista_int)):
        #soma o numero de pontos do intervalo com todos pontos ja somados
        soma+=lista_int[i]
        #divide a soma pelo número de pontos
        x_t=soma/len(pontos)
        #adiciona a lista da cumulada
        lista_cumulada.append(x_t)
    return lista_cumulada

#calcula número de pontos necessários
def Cal_n_k():
    #k obtido por 1/erro 
    k=2000
    #variancia máxima da binomial
    var=0.25
    #calcula o n necessário e transforma em int 
    n=((1.95)**2)*var/0.0005**2
    n=int(n)
    return n,k

#função responsável por gerar gráfico
def graf(pontos,g):
    #gera listas
    lista_u=[]
    lista_v1=[]
    lista_v=[]
    #calcula o intervalo entre os pontos 
    v=pontos[-1]/g
    #loop que gera uma lista como todos os intervalos
    for i in range(g+1):
        lista_v1.append(v*(i))
    #Calcula número de pontos entre intervalos
    lista_v,z=np.histogram(pontos,bins=lista_v1)
    soma=0
    #pega o número de pontos entre intervalos, soma ao total de pontos já percorridos, divide a soma pelo numero
    #total de pontos e adiciona na lista dos valores de U(v)
    lista_u.append(0)
    for j in range(len(lista_v)):
        soma+=lista_v[j]
        lista_u.append(soma/len(pontos))
    #gera a imagem
    plt.axis([-0.2,lista_v1[-1]+0.1*lista_v1[-1],0,lista_u[-1]+0.05])
    plt.plot(lista_v1,lista_u,'r')
    #linhas que representam os valores da cumulada para pontos abaixo do menor valor e acima do maior valor dos pontosd
    plt.hlines(y=0,xmax=pontos[0],xmin=-2,color='r')
    plt.hlines(y=1,color='r',xmin=pontos[-1],xmax=1000)
    plt.xlabel('v')
    plt.ylabel('U(v)')
    plt.title('Valores de U(v) utilizando %i pontos' %g)
    plt.show()
    
def main():
    #pede valores de x
    while True:
        numeros_x=[]
        try:
            x=str(input("Digite a vetor de x (Exemplo: 1 2 3): "))
            numeros_x=re.findall('\d+',x)
        except:
            print("Valores inválidos. Tente novamente")
        else:
            if len(numeros_x)!=3:
                print("Digite 3 valores")
            else:
                break
    #pede valores de y
    while True:
        numeros_y=[]
        try:
            y=str(input("Digite a vetor de y (Exemplo: 1 2 3): "))
            numeros_y=re.findall('\d+',y)
        except:
            print("Valores inválidos. Tente novamente")
        else:
            if len(numeros_y)!=3:
                print("Digite 3 valores")
            else:
                break
    #pede valor da seed
    while True:
        try:
            seed=int(input("Digite a seed: "))
        except:
            print("Valor inválido")
        else:
            break
    np.random.seed(seed)
    #calcula o A
    A=Cal_A(numeros_x,numeros_y)
    #calcula o número de partições(k) e o número de pontos necessários(n)
    n,k=Cal_n_k()
    #calcula os n pontos gerados pela f(theta|A) e devolve a lista ordenada
    pontos=Cal_T(n, A)
    #calcula o valor dos intervalos e gera um vetor com todos os intervalos
    kpontos = []
    intervalo = (pontos[-1] - pontos[0])/k
    for i in range(k+1):
        kpontos.append(pontos[0]+(intervalo*i))
    print("O valor máximo de f(theta|A)=",pontos[-1])
    #Teste de um determinado v
    while True:
        try:
            v=float(input("Digite um inteiro para calcular U(v) ou digite uma letra para finalizar: "))
        except:
            break
        else:
            cumu=Cal_v(pontos,v)
            print(f"U({v}) = {cumu}")
    #Pergunta se deseja ver a lista das cumuladas
    while True:
        try:
            v=int(input("Deseja ver a cumulada entre cada intervalo? Sim=1 Não=2: "))
        except:
            print("Digite 1 ou 2")
        else:
            if v==1:
                #gera lista com a cumulada entre cada intervalo
                lista = salto(pontos, kpontos)
                for i in range(0,len(lista),2):
                    print(f'{f"U({kpontos[i+1]})":<24} = {f"{lista[i]}":<23}  |     {f"U({kpontos[i+2]})":<24} = {f"{lista[i+1]}":<23}')
                break
            if v==2:
                break
            print("Valor inválido")
    #pergunta ao usuário se deseja fazer o gráfico
    while True:
        try:
            x=int(input("Deseja gerar os gráficos dos U(v)? Sim=1 Não=2 :"))
        except:
            print("Digite 1 ou 2")
        else:
            if x==1:
                try:
                    #obtém numero de pontos
                    g=int(input("Digite o número de pontos usados: "))
                except:
                    print("Digite um valor válido")
                else:
                    graf(pontos,g)
                    break
            if x==2:
                break
main()
