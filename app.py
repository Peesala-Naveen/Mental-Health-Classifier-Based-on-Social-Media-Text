import streamlit as st
from utils.predict import predict, explain
import streamlit.components.v1 as components

st.set_page_config(page_title="Mental Health Analyzer")

st.title("🧠 Mental Health Text Analyzer")

# INPUT
input_text = st.text_area("Enter your text:")

# BUTTON
if st.button("Analyze"):

    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        label, probs, idx, class_names = predict(input_text)

        # RESULT
        st.subheader("Prediction")
        st.success(label)

        st.subheader("Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {probs[i]:.4f}")

        # LIME
        st.subheader("LIME Explanation")

        html = explain(input_text)
        components.html(html, height=650, scrolling=True)

# KEEP APP ALIVE (IMPORTANT)
st.write("App is running...")




# venv\Scripts\activate
# python -m streamlit run app.py