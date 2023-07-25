import scipy.special as stats
import numpy as np
import re
import matplotlib.pyplot as plt
import time

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

#Calcular o potencial de uma dirichlet
def potencial(theta,A,gamma):
    prod=1
    for z in range(3):
        prod*=theta[z]**(int(A[z])-1)
    #multiplica o valor gerado na multiplicação por gamma
    prod=prod*gamma
    return prod

#"Aquece" o valor de theta a partir de um theta inicial e matriz de variância fixada
def aquecer(n_inicial,A,gamma):
    #declara variáveis
    theta=[1/3,1/3,1/3] #armazena o valor atual de theta
    thetanovo=[0,0,0] #armazena o candidato para proximo valor de theta
    n_passos=0 #contador de passos
    tentativas=0 #contador global
    soma_theta=0
    while n_passos <= n_inicial:
        soma_theta=0
        tentativas+=1
        y=np.random.multivariate_normal(theta,[[0.03,0,0],[0,0.03,0],[0,0,0.03]]) #gera os pontos da normal multivariada
        for g in range(3): #corrige os valores para que não sejam negativos
            thetanovo[g]=abs(y[g])
            soma_theta+=thetanovo[g]
        for g in range(3): #corrige os valores para que eles somem 1
          thetanovo[g]=thetanovo[g]/soma_theta
        alpha=potencial(thetanovo,A,gamma)/potencial(theta,A,gamma) #calcula alpha, que é a probabilidade de aceitação
        if alpha >=1:
            theta=thetanovo[:]
            n_passos+=1
        else: #aceita com probabilidade alpha, caso ele seja menor que 1
            uniforme=np.random.random()
            if alpha>uniforme:
                theta=thetanovo[:]
                n_passos+=1
            else:
                continue
    return theta


#gera os valores de t(theta|A)
def Cal_T(n,A,n_inicial,cons):
    #calcula o valor da gamma de A
    gamma=const_norm(A)
    #craindo variáveis responsáveis
    prod=1
    n_passos=0
    tentativa=0
    pode=0 #se o novo valor de theta for aceito, vale 1. Caso contrário, vale 0
    mudanca=0 #quando não for necessário calcular a variância, assume valor de 1
    igual=0
    #aquece os thetas
    theta = aquecer(n_inicial,A,gamma)
    #cria as matrizes de covariancia inicial e a sigma, além dos vetores
    #responsáveis por armazenar os pontos obtidos e o valor dos thetas
    matriz_inicial = [[0.03,0,0],[0,0.03,0],[0,0,0.03]]
    sigma = [[0.03,0,0],[0,0.03,0],[0,0,0.03]]
    thetanovo=[0,0,0]
    pontos_diri=[]
    sigma2=[1,1,1]
    t=[[theta[0]],[theta[1]],[theta[2]]]
    #loop principal do programa responsável por criar n thetas permitidos
    #e calcular a produtória desse theta
    while n_passos <=n:
        #reseta as variáveis
        tentativa+=1
        soma_theta=0
        #Após o primeiro theta, é armazenado o theta para o calculo da variancia, caso ainda seja necessário calculá-la
        if n_passos!=0 and pode==1 and mudanca==0:
            t[0].append(theta[0])
            t[1].append(theta[1])
            t[2].append(theta[2])
        #A cada x passos, o valor da variância é calculado, se necessário
        if n_passos%5000==0 and n_passos!=0 and mudanca==0 and pode==1:
            for i in range(3):
                sigma[i][i]=cons*(matriz_inicial[i][i])+(1-cons)*np.var(t[i])
                #se a variância não mudar mais do que 10^-6 da iteração anterior para essa, para de calcular a variância até o fim da execução
                if abs(sigma[i][i]-sigma2[i])<=10**(-6):
                    igual+=1
                sigma2[i]=sigma[i][i]
            if igual==3:
                mudanca=1
                igual=0
                parada=n_passos
            else:
                igual=0
        #calcula, corrige e avalia o valor do candidato ao novo theta
        pode=0
        y=np.random.multivariate_normal(theta,sigma)
        for g in range(3):
            thetanovo[g]=abs(y[g])
            soma_theta+=thetanovo[g]
        for g in range(3):
          thetanovo[g]=thetanovo[g]/soma_theta
        alpha=potencial(thetanovo,A,gamma)/potencial(theta,A,gamma)
        if alpha >=1:
            theta=thetanovo[:]
            n_passos+=1
            pode=1
        else:
            uniforme=np.random.random()
            if alpha>uniforme:
                theta=thetanovo[:]
                n_passos+=1
                pode=1
            else:
                continue
        pontos_diri.append(potencial(theta,A,gamma))
    print("O ponto de parada da mudança da matriz de variancia foi no passo:",parada)
    print("taxa de rejeição",float((tentativa-n_passos)/tentativa))
    pontos_diri.sort()
    return pontos_diri

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

#gera amostra com tamanho n usando o gerador numpy da dirichlet
def numpy(n,A):
    lista_diri_gerador=[]
    gamma=const_norm(A)
    y=np.random.dirichlet(A,n)
    prod=1
    for n in range(n):
        for k in range(3):
            prod*=y[n][k]**(int(A[k])-1)
        #multiplica o valor gerado na multiplicação por gamma
        prod=prod*gamma
        #adiciona ao vetor de pontos
        lista_diri_gerador.append(prod)
        #reseta a multiplicação
        prod=1
    lista_diri_gerador.sort()
    return lista_diri_gerador

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
def graf(pontos,k,lista_diri_gerador):
    #gera listas para os pontos da MCMC
    lista_u=[]
    lista_v1=[]
    lista_v=[]
    #gera lista dos pontos da numpy
    lista_diri_partições=[]
    lista_diri_n_particao=[]
    lista_diri_w=[]
    #calcula o intervalo entre os pontos
    v=pontos[-1]/k
    v2=lista_diri_gerador[-1]/k
    #loop que gera uma lista como todos os intervalos
    for i in range(k+1):
        lista_v1.append(v*(i))
        lista_diri_partições.append(v2*(i))
    #Calcula número de pontos entre intervalos
    lista_v,z=np.histogram(pontos,bins=lista_v1)
    lista_diri_n_particao,z=np.histogram(lista_diri_gerador,bins=lista_diri_partições)
    soma1=0
    soma2=0
    #pega o número de pontos entre intervalos, soma ao total de pontos já percorridos, divide a soma pelo numero
    #total de pontos e adiciona na lista dos valores de U(v)
    lista_u.append(0)
    lista_diri_w.append(0)
    for j in range(len(lista_v)):
        soma1+=lista_v[j]
        lista_u.append(soma1/len(pontos))
        soma2+=lista_diri_n_particao[j]
        lista_diri_w.append(soma2/len(lista_diri_gerador))
    #gera a imagem
    plt.axis([-0.2,lista_v1[-1]+0.1*lista_v1[-1],0,lista_u[-1]+0.05])
    plt.plot(lista_v1,lista_u,'r',label='Gerador MCMC')
    plt.plot(lista_diri_partições,lista_diri_w,'b',label='Gerador numpy')
    #linhas que representam os valores da cumulada para pontos abaixo do menor valor e acima do maior valor dos pontos
    plt.hlines(y=0,xmax=pontos[0],xmin=-2,color='r')
    plt.hlines(y=0,xmax=lista_diri_gerador[0],xmin=-2,color='b')
    plt.hlines(y=1,color='r',xmin=pontos[-1],xmax=1000)
    plt.hlines(y=1,color='b',xmin=lista_diri_gerador[-1],xmax=1000)
    plt.xlabel('v')
    plt.ylabel('U(v)')
    plt.title('Valores de U(v) utilizando %i intervalos' %k)
    plt.legend(loc='upper left')
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
    #pede o número de pontos para aquecer o theta
    while True:
        try:
            n_inicial=int(input("Digite o número de pontos para aquecer a cadeia: "))
        except:
            print("Valor inválido")
        else:
            break
    #pede o valor da constante
    while True:
        try:
            cons=float(input("Digite o valor da constante: "))
        except:
            print("Valor inválido")
        else:
            break
    #calcula o A
    A=Cal_A(numeros_x,numeros_y)
    #calcula o número de partições(k) e o número de pontos necessários(n)
    n,k=Cal_n_k()
    #calcula os n pontos gerados pela f(theta|A) e devolve a lista ordenada
    start_time = time.time()
    pontos=Cal_T(n, A,n_inicial,cons)
    pontos_numpy=numpy(n,A)
    print("o tempo necessário foi de ",time.time()-start_time)
    #calcula o valor dos intervalos e gera um vetor com todos os intervalos
    kpontos = []
    kpontos_numpy=[]
    intervalo = (pontos[-1] - pontos[0])/k
    intervalo_numpy=(pontos_numpy[-1]-pontos_numpy[0])/k
    for i in range(k+1):
        kpontos.append(pontos[0]+(intervalo*i))
        kpontos_numpy.append(pontos_numpy[0]+(intervalo_numpy*i))
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
    #pergunta ao usuário se ele deseja verificar a variação em cada intervalo do gerador MCMC e o gerador numpy
    while True:
        try:
            v=int(input("Deseja ver a variação da cumulada entre os dois geradores em cada intervalo ? Sim=1 Não=2: "))
        except:
            print("Digite 1 ou 2")
        else:
            if v==1:
                #gera lista com a cumulada entre cada intervalo para ambos os geradores
                lista = salto(pontos, kpontos)
                lista_numpy=salto(pontos_numpy,kpontos_numpy)
                for i in range(0,len(lista),2):
                    print(f'{f"U({kpontos_numpy[i+1]})":<24} = {f"{abs(lista[i]-lista_numpy[i])}":<23}  |     {f"U({kpontos_numpy[i+2]})":<24} = {f"{abs(lista[i+1]-lista_numpy[i+1])}":<23}')
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
                    graf(pontos,g,pontos_numpy)
                    break
            if x==2:
                break
main()