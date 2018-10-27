import pandas as pd

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)

X = Xdummies_df.values
Y = Y_df.values

porcentagem_teino = 0.9
tamanho_treino = int(porcentagem_teino * len(Y))
tamanho_teste = len(Y) - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)
total_elementos = len(teste_dados)
taxa_acerto = total_de_acertos/total_elementos * 100.0

print(taxa_acerto)
print(total_elementos)