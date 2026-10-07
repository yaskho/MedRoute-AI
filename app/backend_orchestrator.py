import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
import json
from tensorflow.keras import layers
from safetensors.tensorflow import load_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_DIR = "models"

MODEL_CHECKPOINT = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
vocab_size = tokenizer.vocab_size
max_length = tokenizer.model_max_length

print("Chargement des modèles MedRoute AI...")


model_dir = "models/transformer/final_transformer_model"

transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    

rf_model = joblib.load(os.path.join(MODEL_DIR, "diabetes_random_forest.pkl"))
melanoma_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "final_cnn_model.h5"))
print("Modèles chargés avec succès !")

def medroute_orchestrator(prompt, tabular_features=None, image_path=None):
    print(f"\n--- Requête reçue : '{prompt}' ---")
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",     
        padding=True,
        truncation=True,
        max_length=128
    )

    transformer_model.eval()
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits
        
    probs = torch.softmax(logits, dim=-1).tolist()[0]
    intent_idx = np.argmax(probs)
    
    intent_mapping = {0: "TABULAR_ONLY", 1: "IMAGE_ONLY", 2: "MULTIMODAL"}
    selected_path = intent_mapping[intent_idx]
    print(f"[Routeur AI] Chemin sélectionné --> {selected_path}")
    
    
    return selected_path


if __name__ == "__main__":
    sample_text = "analyser le risque de diabete a partir de glucose"

    cls, probs = medroute_orchestrator(sample_text)
    breakpoint()
    print("Résultat final :", res)
