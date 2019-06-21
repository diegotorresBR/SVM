import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from math import *
import scipy.optimize as oti


def main():
    dim = 2 #Dimensão do Espaço
    numero_de_pontos = 400 #Numero de Pontos Para o Aprendizado

    #Intervalo dos Pontos gerados
    l_inf = -100
    l_sup = 100

    def classifica(v):
        a = 1.0
        b = -1.0
        c = 0.0
        delta = 10
        if np.dot(np.array([a,b]), v) + c > delta:
            return 1
        if np.dot(np.array([a,b]), v) + c < delta:
            return -1
        return 0


    vetores = [ponto_aleatorio(dim, l_inf, l_sup) for x in range(numero_de_pontos)]
    vetores = [v for v in vetores if classifica(v) != 0]
    vet_aleatorio = [classifica(v) for v in vetores]

    w, b = perceptron(vetores, vet_aleatorio)
    #ac, ab = classificar(w, b, vetores, dim)
    ac = [v for v,c in zip(vetores, vet_aleatorio) if c == 1]
    ab = [v for v,c in zip(vetores, vet_aleatorio) if c == -1]
    plotar(ac, ab, w, b, l_inf, l_sup, dim)


def otimizar(w, b):
    restr = ({'type': 'ineq', 'fun': lambda x: w + b -1})
    oti.minimize(f_objetivo())

def f_objetivo(w, b):
    return 1/2*norma(w)**2-alpha*(w + b -1)




def perceptron(vetores, vet_classifica):
    w_vetor = np.zeros(np.shape(vetores[0]), dtype=float)
    b_vies = 0.0
    erros = 0
    r = max(norma(x) for x in vetores)
    print("Norma", r)
    r *= r # r ao quadrado

    taxa = 0.5#Taxa de Aprendizado
    e = 200
    while(e > erros):
        for x, y in zip(vetores, vet_classifica):
            if y*(np.dot(w_vetor, x) + b_vies)<=0:
                w_vetor += taxa * x * y
                b_vies += taxa * y * r
                erros = erros + 1
        e-=1
    print("Vies", b_vies)
    return w_vetor, b_vies

# def perceptron2(dim_vetor, qtd_pontos, vetores, vet_classifica):
#     alpha = 0
#     w_vetor = zeros(dim_vetor, dtype=float)
#     b_vies = 0
#     erros = 0
#     r = max(norma(x) for x in vetores)
#     print("Norma", r)
#
#     taxa = 0.5  # Taxa de Aprendizado
#
#     for x in range(qtd_pontos):
#         if (vet_classifica[x] * (dot(w_vetor, vetores[x]) + b_vies) <= 0):
#             w_vetor = w_vetor + taxa * dot((vet_classifica[x]), vetores[x])
#             # b_vies = b_vies + (taxa * vet_classifica[x] * (r ** 2))
#             erros = erros + 1
#     print(erros, "erros")
#     # print(w_vetor)
#     print("Vies", b_vies)
#     return w_vetor, b_vies




def norma(vetor):
    norma = 0
    for x in vetor:
        norma  = norma + x**2
    norma = sqrt(norma)
    return norma

def ponto_aleatorio(dimensao, l_i, l_s):
    ponto = np.zeros(dimensao, dtype=float)
    for x in range(dimensao):
        n_aleatorio = rnd.randrange(l_i, l_s)#uniform #randrange
        ponto[x] = n_aleatorio
    return ponto

def classificar(w, b, dados, dimensao):
    acima = []
    abaixo = []

    for x in dados:
        prod = np.dot(w,x) + b
        if(prod >=0):
            acima.append(x)
        else:
            abaixo.append(x)
    eq_hp = ""
    for x in range(dimensao):
        eq_hp = eq_hp + str(w[x]) + "x"+str(x) + " "
        eq_hp_ = eq_hp + str(b)
    print("Hiperplano", eq_hp_)

    return acima, abaixo

def plotar(acima, abaixo, hiper, vies, lim_abaixo, lim_acima, dimensao):
    fig, ax = plt.subplots()

    ax.plot([lim_abaixo, lim_acima], [lim_abaixo, lim_acima], 'g-')

    ax.plot([x[0] for x in acima], [x[1] for x in acima], '.')
    ax.plot([x[0] for x in abaixo], [x[1] for x in abaixo], 'x')

    ##Criar Reta

    # reta = []
    # pontos = pontos_hiperplano(vies, hiper, lim_abaixo, lim_acima)
    # reta.append(pontos[0])
    # reta.append(pontos[1])

    d1_max = min(distancia(hiper, x, vies) for x in acima)
    l_dist = [x for x in acima if distancia(hiper, x, vies) == d1_max]

    d2_max = min(distancia(hiper, x, vies) for x in abaixo)
    l_dist2 = [x for x in abaixo if distancia(hiper, x, vies) == d2_max]


    ax.plot(l_dist[0][0], l_dist[0][1], "bo")
    ax.plot(l_dist2[0][0], l_dist2[0][1], "yo")

    vies2 = new_bias(vies, hiper, l_dist2[0])
    x,y = pontos_hiperplano(vies2, hiper, lim_abaixo, lim_acima)
    ax.plot(x, y, 'y--')

    vies1 = new_bias(vies, hiper, l_dist[0])
    x,y = pontos_hiperplano(vies1, hiper, lim_abaixo, lim_acima)
    ax.plot(x, y, 'b--')

    x,y = pontos_hiperplano((vies1+vies2)/2, hiper, lim_abaixo, lim_acima)
    ax.plot(x, y, 'k-')

    plt.show()

def pontos_hiperplano(vies, hiper, lim_inf, lim_sup):
    #Apenas para construir a reta
    n1 = (-vies - (lim_inf * hiper[0]))/hiper[1]
    n2 = (-vies - (lim_sup * hiper[0]))/hiper[1]

    return [lim_inf, lim_sup], [n1, n2]

def distancia(hiper, ponto, vies):
    d = abs((hiper[0]*ponto[0]) + (hiper[1]*ponto[1]) + vies) / norma(hiper)
    return d

def new_bias(vies, hiper, P):
    print(hiper.shape, P.shape)
    return -np.dot(hiper, P)

if __name__ == "__main__":

   main()