import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
import os
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X3 = np.load('Banco/X9.npy')
y3 = np.load('Banco/y9.npy')

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3, random_state=42)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
knn = KNeighborsClassifier()
parametros = {'n_neighbors': [3, 5, 7, 9, 11]}

grid = GridSearchCV(knn, parametros, cv=skf)

grid.fit(X_train, y_train)

print("Melhores hiperparâmetros: ", grid.best_params_)
print("Melhor score: ", grid.best_score_)

# # Teste final com os 30% separados
# y_pred = grid.predict(X_test)
# score = grid.score(X_test, y_test)
# print("Acurácia no conjunto de teste:", score)

# ...

# realiza a validação cruzada com todos os valores de n_neighbors
cv_results = grid.cv_results_

for i in range(len(parametros['n_neighbors'])):
    mean_test_score = cv_results[f'mean_test_score'][i]
    std_test_score = cv_results[f'std_test_score'][i]
    k = parametros['n_neighbors'][i]
    
    # treina o modelo com o melhor valor de K
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # realiza as previsões no conjunto de teste
    y_pred = knn.predict(X_test)
    
    # calcula as métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # imprime os resultados
    print(f'K = {k}:')
    print(f'Acurácia: {accuracy:.4f}')
    print(f'Precisão: {precision:.4f}')
    print(f'Revocação: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
