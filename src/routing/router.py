import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def classify_intent(prompt, transformer_model, tokenizer, max_length=20):
    """
    Analyse une requête textuelle et détermine le chemin de routage.
    """
    seq = tokenizer.texts_to_sequences([prompt])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    prediction = transformer_model.predict(padded_seq, verbose=0)
    intent_idx = np.argmax(prediction)
    
    intent_mapping = {0: "TABULAR_ONLY", 1: "IMAGE_ONLY", 2: "MULTIMODAL"}
    return intent_mapping.get(intent_idx, "UNKNOWN")