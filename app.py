import streamlit as st
import numpy as np
from lime.lime_text import LimeTextExplainer
from utils.predict import predict, predict_proba_func

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Mental Health Classifier", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>

        html, body, .main {
            background-color: #fff !important;
            color: #000 !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #000 !important;
        }

        /* Transparent Text Area */
        .stTextArea textarea {
            background-color: transparent !important;
            color: #000 !important;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
            border: 1px solid #bbb;
            box-shadow: none;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #ffb347, #ffcc33);
            color: #000;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #ffcc33, #ffb347);
            transform: scale(1.05);
        }

        /* Success Box */
        .stAlert {
            border-radius: 10px;
            color: #000 !important;
            background: #e6ffe6 !important;
        }

        /* Card Style */
        .card {
            background-color: #f5f5f5 !important;
            padding: 20px;
            border-radius: 15px;
            margin-top: 15px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
            color: #000 !important;
        }

        /* Probabilities */
        .prob-box {
            background-color: #f5f5f5 !important;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 8px;
            color: #000 !important;
        }

        /* Hide scrollbars globally */
        ::-webkit-scrollbar {
            width: 0px;
            height: 0px;
            background: transparent;
            display:hidden;
        }
        /* Hide scrollbars for all browsers */
        html {
            scrollbar-width: none;
            -ms-overflow-style: none;
        }
        html::-webkit-scrollbar, body::-webkit-scrollbar, .main::-webkit-scrollbar {
            display: none;
        }

        /* LIME Explanation override */
        .lime, .lime *, .lime table, .lime tr, .lime td, .lime th, .lime span, .lime div, .lime p, .lime b, .lime strong {
            color: #000 !important;
            background: #fff !important;
        }
        /* Hide scrollbars inside LIME explanation */
        .lime::-webkit-scrollbar {
            width: 0px;
            height: 0px;
            background: transparent;
            display: none;
        }
        .lime {
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<h1>🧠 Mental Health Text Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#aaa;'>Enter text to analyze mental health condition</p>", unsafe_allow_html=True)



# ------------------ INPUT ------------------
input_text = st.text_area("Enter your text here:")

# ------------------ BUTTON ------------------
if st.button("🚀 Predict"):

    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text")

    else:
        try:
            pred_class, probs, idx, class_names = predict(input_text)

            # ------------------ PREDICTION ------------------
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔍 Prediction")
            st.success(f"Predicted Class: **{pred_class}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # ------------------ PROBABILITIES ------------------
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Probabilities")

            for i, c in enumerate(class_names):
                st.markdown(
                    f"<div class='prob-box'>🔹 <b>{c}</b>: {probs[i]:.4f}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # ------------------ LIME ------------------
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧪 LIME Explanation")

            explainer = LimeTextExplainer(class_names=class_names)

            explanation = explainer.explain_instance(
                input_text,
                predict_proba_func,
                num_features=10,
                labels=[idx]
            )

            html = explanation.as_html()

            # Wrap LIME HTML in a div with class 'lime' to force white text
            st.components.v1.html(f"<div class='lime'>" + html + "</div>", height=500, scrolling=True)

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")