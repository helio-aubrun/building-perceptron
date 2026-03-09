from perceptron import Perceptron
from cleaner import load_and_clean_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Obtenir les données propres
X_train, X_test, y_train, y_test = load_and_clean_data('bcw_data.csv')

# 2. Initialiser et entraîner le Perceptron
model = Perceptron(learning_rate=0.01, n_epochs=100)
model.fit(X_train, y_train)

# 3. Prédictions
predictions = model.predict(X_test)

# 4. Évaluation des résultats
print("--- RÉSULTATS DE LA MODÉLISATION ---")
print(f"Précision Globale : {accuracy_score(y_test, predictions):.2%}")
print("\nTableau détaillé des performances :")
# On utilise target_names pour plus de clarté dans le rapport de santé
print(classification_report(y_test, predictions, target_names=['Bénin (B)', 'Malin (M)']))

# Matrice de confusion pour voir les erreurs de diagnostic
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prédit B', 'Prédit M'], 
            yticklabels=['Réel B', 'Réel M'])
plt.title('Matrice de Confusion : Diagnostic du Cancer')
plt.show()