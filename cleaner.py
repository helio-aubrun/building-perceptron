import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    # Chargement
    df = pd.read_csv(filepath)
    
    # Nettoyage : suppression de l'ID et des colonnes vides
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.dropna(axis=1, how='all') # Supprime les colonnes vides (ex: Unnamed: 32)

    # Préparation des features et de la cible
    # IMPORTANT : M -> 1, B -> -1 pour correspondre à votre fonction d'activation
    y = df['diagnosis'].map({'M': 1, 'B': -1}).values
    X = df.drop(columns=['diagnosis']).values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation : Crucial pour la convergence du Perceptron
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test