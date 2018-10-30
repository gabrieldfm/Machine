import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import cross_val_score

df = pd.read_csv('situacao_cliente.csv')
X_df = df[['recencia','frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)

    taxa_acerto = np.mean(scores)

    msg = "Taxa de acertodo {0}: {1}".format(nome, taxa_acerto)
    print(msg)
    return taxa_acerto

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVRest = OneVsRestClassifier(LinearSVC(random_state =  0))
resultadoOneVRest = fit_and_predict("OneVesusRest", modeloOneVRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVRest] = modeloOneVRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVesusOne = OneVsOneClassifier(LinearSVC(random_state =  0))
resultadoOneVesusOne = fit_and_predict("OneVesusOne", modeloOneVesusOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVesusOne] = modeloOneVesusOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

print(resultados)
maximo = max(resultados)
vencedor = resultados[maximo]
print("Vencerdor: ")
print(vencedor)
