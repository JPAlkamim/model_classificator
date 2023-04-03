from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import numpy as np

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

svm = SVC()

param_grid = [    
    {"kernel": ['rbf'], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "C": [0.1, 1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]

gscv = GridSearchCV(svm, param_grid, cv=skf, scoring='f1_macro')
print("Aqui")
gscv.fit(X_train, y_train) 

print("Best parameters set found on development set:")
print(gscv.best_params_)

print('Mean cross-validation score: {:.5f}'.format(gscv.best_score_))

svm_best = gscv.best_estimator_

svm_best.fit(X_train, y_train)

y_pred = svm_best.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print('Test set accuracy: {:.5f}'.format(svm_best.score(X_test, y_test)))
print('Test set F1-Score: {:.5f}'.format(f1))
