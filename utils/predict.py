import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text
from utils.capsule_layer import CapsuleLayer    # ✅ ADD THIS

MODEL_PATH = "model/CONV_BILSTM_CAPS_Dynamic_Model.keras"
TOKENIZER_PATH = "model/CONV_BILSTM_CAPS_Dynamic_tokenizer.pkl"
ENCODER_PATH = "model/CONV_BILSTM_CAPS_Dynamic_encoder.pkl"

# ✅ LOAD MODEL WITH CUSTOM LAYER
model = load_model(
    MODEL_PATH,
    custom_objects={"CapsuleLayer": CapsuleLayer},
    compile=False
)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

MAX_LEN = 150

def predict_proba_func(texts):
    clean_texts = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(clean_texts)
       # 🔥 FORCE correct input size
    padded = pad_sequences(
        sequences,
        maxlen=150,
        padding='post',
        truncating='post'
    )
    return model.predict(padded)

def predict(text):
    probs = predict_proba_func([text])[0]
    idx = np.argmax(probs)
    return le.classes_[idx], probs, idx, le.classes_