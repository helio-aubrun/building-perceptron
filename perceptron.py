import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        Constructeur du perceptron
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0

    def activation(self, z):
        """
        Fonction d'activation (fonction seuil)
        """
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        """
        Entraînement du perceptron
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)

                # Mise à jour des poids
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        """
        Prédiction pour de nouvelles données
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)