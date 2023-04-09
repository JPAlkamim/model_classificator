from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

# remover todas as ocorrências das classes 1 e 4
X = X[~np.isin(y, [1, 4])]
y = y[~np.isin(y, [1, 4])]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_scores = []
f1_scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svc = SVC(kernel='rbf', gamma=1, C=1)

    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    acc_scores.append(acc)
    f1_scores.append(f1)

print("Acurácia média: ", np.mean(acc_scores))
print("F1-score médio: ", np.mean(f1_scores))
