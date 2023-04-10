from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix, binary_confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

y[y == 4] = 5

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_scores = []
f1_scores = []
binary_conf_mats = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm = SVC(kernel='rbf', gamma=1, C=1000)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_test[y_test == 4] = 5
    y_pred[y_pred == 4] = 5

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    binary_conf_mat = binary_confusion_matrix(y_test, y_pred, labels=[4, 5])
    
    acc_scores.append(acc)
    f1_scores.append(f1)
    binary_conf_mats.append(binary_conf_mat)

print("Acurácia média: ", np.mean(acc_scores))
print("F1-score médio: ", np.mean(f1_scores))

# Obter a melhor matriz de confusão binária
best_conf_mat = binary_conf_mats[np.argmax(acc_scores)]

# Definir as classes
classes = ['classe 4', 'outras']

# Plotar a matriz de confusão da classe 4
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(best_conf_mat, cmap='Blues')

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
ax.set_title("Matriz de confusão para classe 4")

# Mostrar o plot
plt.show()
