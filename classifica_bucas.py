from collections import Counter
import pandas as pd

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)

X = Xdummies_df.values
Y = Y_df.values

porcentagem_teino = 0.8
porcentagem_teste = 0.1

tamanho_treino = int(porcentagem_teino * len(Y))
tamanho_teste = int(porcentagem_teste * len(Y))
tamanho_validacao = len(Y) - tamanho_treino - tamanho_teste

treino_dados = X[0:tamanho_treino]
treino_marcacoes = Y[0 :tamanho_treino]

fim_teste = tamanho_treino + tamanho_teste
teste_dados = X[tamanho_treino:fim_teste]
teste_marcacoes = Y[tamanho_treino:fim_teste]

validacao_dados = X[fim_teste:]
validacao_marcacoes = Y[fim_teste:]

def fit_and_predict(nome, modelo,treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes

    acertos = [d for d in diferencas if d == 0]

    total_de_acertos = len(acertos)
    total_elementos = len(teste_dados)
    taxa_acerto = total_de_acertos/total_elementos * 100.0

    print("Taxa de acerto do {0}: {1}".format(nome, taxa_acerto))

    acerto_base = max(Counter(teste_marcacoes).values())
    taxa_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
    print("Taxa acerto base: %f" % taxa_acerto_base)
    return taxa_acerto

from sklearn.naive_bayes import MultinomialNB
modelo_multi = MultinomialNB()
resultado_multi = fit_and_predict("MultinomialNB",modelo_multi, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes )

from sklearn.ensemble import AdaBoostClassifier
modelo_ada = AdaBoostClassifier()
resultado_ada = fit_and_predict("AdaBoostClassifier", modelo_ada, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes )

print(len(teste_dados))

