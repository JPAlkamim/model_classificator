from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import numpy as np

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definindo o modelo
rf = RandomForestClassifier()

# Definindo os parâmetros que serão testados
param_grid = {
    'max_depth': [10, 50],
    'min_samples_leaf': [1, 3],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 200]
}

# Realizando a busca pelos melhores parâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=skf)
print("Aqui")
grid_search.fit(X_train, y_train)

# Imprimindo os resultados
print("Best parameters set found on development set:")
print(grid_search.best_params_)
print('Mean cross-validation score: {:.5f}'.format(grid_search.best_score_))

# Obtendo o modelo com os melhores parâmetros
best_rf = grid_search.best_estimator_

# Fazendo a predição no conjunto de teste
y_pred = best_rf.predict(X_test)

# Avaliando o desempenho do modelo
print('Test set F1-score: {:.5f}'.format(f1_score(y_test, y_pred, average='macro')))
print('Test set accuracy: {:.5f}'.format(best_rf.score(X_test, y_test)))