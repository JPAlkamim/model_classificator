import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import os
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X3 = np.load('Banco/X9.npy')
y3 = np.load('Banco/y9.npy')

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
parametros = {'n_neighbors': [3, 5, 7, 9, 11]}

grid = GridSearchCV(knn, parametros, cv=10)

grid.fit(X_train, y_train)

print("Melhores hiperparâmetros: ", grid.best_params_)
print("Melhor score: ", grid.best_score_)

# Teste final com os 30% separados
y_pred = grid.predict(X_test)
score = grid.score(X_test, y_test)
print("Acurácia no conjunto de teste:", score)