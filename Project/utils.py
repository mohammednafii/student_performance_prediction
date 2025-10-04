import pandas as pd
import pickle
    
def load_model():
    with open(r"C:\Users\moham\pfe\random_forest_cluster.pkl", 'rb') as f:
        return pickle.load(f)
def load_scaler():
    with open(r"C:\Users\moham\pfe\scaler2.pkl", 'rb') as f:
        return pickle.load(f)

def load_database():
    return pd.read_csv("etudiant_performance.csv")
def load_model_formulaire():
    with open(r"C:\Users\moham\pfe\random_forest.pkl", 'rb') as f:
        return pickle.load(f)