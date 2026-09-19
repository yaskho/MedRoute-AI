import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os

# --- CHARGEMENT DES MODÈLES ---
# Ajuste le chemin "../models" selon la position de ton fichier main.py dans app/
MODEL_DIR = "../models"

print("Chargement des modèles MedRoute AI...")
rf_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_random_forest.pkl"))
melanoma_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "medroute_melanoma_model.h5"))
print("Modèles chargés avec succès !")

def medroute_orchestrator(prompt, tabular_features=None, image_path=None):
    """
    Orchestrateur central MedRoute AI
    """
    print(f"\n--- Requête reçue : '{prompt}' ---")
    
    # Simulation du routage (ou appel de ton module router.py)
    # Ici on simule l'intention selon les mots clés pour l'exemple global
    if "glycémie" in prompt.lower() or "diabète" in prompt.lower():
        selected_path = "TABULAR_ONLY"
    elif "mélanome" in prompt.lower() or "photo" in prompt.lower():
        selected_path = "IMAGE_ONLY"
    else:
        selected_path = "MULTIMODAL"
        
    print(f"[Routeur AI] Chemin sélectionné --> {selected_path}")
    
    results = {"intent": selected_path}
    
    # Exécution Tabulaire
    if selected_path in ["TABULAR_ONLY", "MULTIMODAL"] and tabular_features is not None:
        tab_pred = rf_model.predict([tabular_features])
        tab_prob = rf_model.predict_proba([tabular_features])
        results["tabular_prediction"] = int(tab_pred[0])
        results["tabular_probability"] = float(np.max(tab_prob))
        print(f"-> [Tabulaire] Risque de diabète : {tab_pred[0]} (Confiance : {np.max(tab_prob)*100:.2f}%)")
        
    # Exécution Imagerie
    if selected_path in ["IMAGE_ONLY", "MULTIMODAL"] and image_path is not None:
        print(f"-> [Imagerie] Analyse de l'image : {image_path}")
        results["image_processed"] = image_path
        
    return results

if __name__ == "__main__":
    # Test global
    sample_features = [63, 130, 60, 23, 190, 25.5, 0.45, 32]
    res = medroute_orchestrator("Analyser la glycémie de ce patient", tabular_features=sample_features)
    print("Résultat final :", res)
