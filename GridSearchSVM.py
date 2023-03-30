from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import numpy as np

X = np.load('Banco/X3.npy')
y = np.load('Banco/y3.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm = SVC()

parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'sigmoid']}

gscv = GridSearchCV(svm, parameters, cv=10)
print("Aqui")
gscv.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(gscv.best_params_)

print('Mean cross-validation score: {:.5f}'.format(gscv.best_score_))

svm_best = gscv.best_estimator_

svm_best.fit(X_train, y_train)

print('Test set accuracy: {:.5f}'.format(svm_best.score(X_test, y_test)))
