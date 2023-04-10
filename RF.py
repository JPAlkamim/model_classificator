from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

# Definir o modelo com os parâmetros especificados
model = RandomForestClassifier(
        max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=200)


# Separar os dados em 10 pastas com a mesma proporção das classes
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Inicializar as listas para armazenar os resultados
acc_list = []
f1_list = []
conf_list = []

# Loop pelas pastas de treino e teste
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer as predições
    y_pred = model.predict(X_test)

    # Calcular a acurácia e f1-score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calcular a matriz de confusão
    classes = np.unique(y)
    conf_mat = confusion_matrix(y_test, y_pred, labels=classes)

    # Adicionar os resultados nas listas
    acc_list.append(acc)
    f1_list.append(f1)
    conf_list.append(conf_mat)

# Calcular a média e desvio padrão da acurácia e f1-score
mean_acc = np.mean(acc_list)
std_acc = np.std(acc_list)

mean_f1 = np.mean(f1_list)
std_f1 = np.std(f1_list)

# Imprimir os resultados
print("Acurácia média: {:.2f} +/- {:.2f}".format(mean_acc, std_acc))
print("F1-score médio: {:.2f} +/- {:.2f}".format(mean_f1, std_f1))

# Encontrar a pasta com a melhor acurácia
best_folder = np.argmax(acc_list)

# Obter a matriz de confusão correspondente
best_conf_mat = conf_list[best_folder]

# Plotar a matriz de confusão da melhor pasta
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(best_conf_mat, cmap='Blues', vmin=0, vmax=20)

# Adicionar as labels dos eixos
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Rotacionar os labels do eixo x
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Adicionar os valores da matriz como anotações
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, best_conf_mat[i, j],
                       ha="center", va="center", color="white")

# Adicionar a barra de cores
cbar = ax.figure.colorbar(im, ax=ax)

# Adicionar o título do plot
ax.set_title("Matriz de confusão da melhor pasta")

# Mostrar o plot
plt.show()



