import numpy as np
from perceptron import Perceptron

np.random.seed(42)

# Nombre d'échantillons
n_samples = 100

# Classe +1
X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y_pos = np.ones(n_samples // 2)

# Classe -1
X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
y_neg = -np.ones(n_samples // 2)

# Données finales
X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

perceptron = Perceptron(learning_rate=0.01, n_epochs=50)

# Entraînement
perceptron.fit(X, y)

# Prédictions
y_pred = perceptron.predict(X)

# Taux de précision
accuracy = np.mean(y_pred == y)
print(f"Précision du modèle : {accuracy:.2f}")