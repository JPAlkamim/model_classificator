from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

# remover todas as ocorrências da classe 4
X = X[y != 4]
y = y[y != 4]

# adicionar um valor ao rótulo da classe 1 para transformá-la em uma nova classe
y[y == 1] = 5

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_scores = []
f1_scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm = SVC(kernel='rbf', gamma=1, C=1000)

    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    # subtrair um valor do rótulo da classe prevista para transformá-la novamente na classe 1
    y_test[y_test == 5] = 1
    y_pred[y_pred == 5] = 1

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    acc_scores.append(acc)
    f1_scores.append(f1)

print("Acurácia média: ", np.mean(acc_scores))
print("F1-score médio: ", np.mean(f1_scores))
