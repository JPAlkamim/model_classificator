import matplotlib.pyplot as plt
import numpy as np
 
# Definindo os resultados para cada modelo
knn_results = [0.8876, 0.8866]
svm_results = [0.76793, 0.76655]
rf_results = [0.86842, 0.86950]
 
# Definindo as posições no eixo x
x_pos = np.arange(len(knn_results))
 
# Criando a figura e os eixos
fig, ax = plt.subplots()
 
# Plotando as barras de acurácia
ax.bar(x_pos, knn_results, width=0.2, color='blue', align='center')
ax.bar(x_pos + 0.2, svm_results, width=0.2, color='red', align='center')
ax.bar(x_pos + 0.4, rf_results, width=0.2, color='green', align='center')
 
# Adicionando as legendas e o nome dos eixos
ax.legend(['KNN', 'SVM', 'Random Forest'])
ax.set_ylabel('Acurácia / F1-Score')
ax.set_xticks(x_pos + 0.2 / 2)
ax.set_xticklabels(['Acurácia', 'F1-Score'])
 
# Mostrando o gráfico
plt.show()
