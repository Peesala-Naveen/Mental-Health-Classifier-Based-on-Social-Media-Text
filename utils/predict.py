import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import clean_text
from utils.capsule_layer import CapsuleLayer

# 🔥 LIME imports
from lime.lime_text import LimeTextExplainer


MODEL_PATH = "model/CONV_BILSTM_CAPS_Dynamic_Model.keras"
TOKENIZER_PATH = "model/CONV_BILSTM_CAPS_Dynamic_tokenizer.pkl"
ENCODER_PATH = "model/CONV_BILSTM_CAPS_Dynamic_encoder.pkl"

# ✅ LOAD MODEL
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

# ------------------ PREDICT PROBA ------------------
def predict_proba_func(texts):
    clean_texts = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(clean_texts)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    return model.predict(padded)

# ------------------ PREDICT ------------------
def predict(text):
    probs = predict_proba_func([text])[0]
    idx = np.argmax(probs)
    return le.classes_[idx], probs, idx, le.classes_

# ------------------ LIME EXPLAIN ------------------
def explain(text):
    class_names = list(le.classes_)

    explainer = LimeTextExplainer(class_names=class_names)

    probs = predict_proba_func([text])[0]
    pred_idx = np.argmax(probs)

    exp = explainer.explain_instance(
        text,
        predict_proba_func,
        num_features=10,
        labels=[pred_idx]
    )

    # 🔥 RETURN HTML (NOT FIGURE)
    html = exp.as_html()
    return html