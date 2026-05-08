# 🧠 Mental Health Classifier Based on Social Media Text

AI-driven Mental Health Detection System using Deep Learning and Capsule Networks for Clinical Support.

---

# 📌 Project Overview

This project is a Deep Learning-based Mental Health Classification System developed using Reddit social media text data.

The system analyzes user-generated text and predicts possible mental health conditions such as:

- Depression
- Anxiety
- Bipolar Disorder
- Borderline Personality Disorder (BPD)
- Schizophrenia
- Autism

The project integrates:

- CNN (Convolutional Neural Network)
- BiLSTM (Bidirectional Long Short-Term Memory)
- Capsule Networks
- LIME Explainability
- GloVe Word Embeddings
- Streamlit Web Application

---

# ✅ Final Model Used

## Conv-BiLSTM-Capsule (Dynamic Routing)

---

# 🚀 Features

- ✅ Mental health prediction from text input
- ✅ Deep learning-based classification
- ✅ LIME explainability visualization
- ✅ Real-time prediction using Streamlit
- ✅ Pre-trained tokenizer and encoder support
- ✅ Capsule Network integration
- ✅ Clean and simple UI

---

# 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Scikit-learn
- NLTK
- LIME
- Capsule Networks

---

# 📂 Project Structure

```text
mental-health-app/
│
├── model/
│   ├── CONV_BILSTM_CAPS_Dynamic_Model.keras
│   ├── CONV_BILSTM_CAPS_Dynamic_tokenizer.pkl
│   └── CONV_BILSTM_CAPS_Dynamic_encoder.pkl
│
├── utils/
│   ├── capsule_layer.py
│   ├── predict.py
│   └── preprocess.py
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
│
├── venv/
└── .venv/
```

---

# ⚙️ Model Workflow

## Step 1: User Input

The user enters mental health-related text into the Streamlit interface.

---

## Step 2: Text Preprocessing

The text is:

- converted to lowercase
- cleaned
- stopwords removed
- tokenized
- padded

---

## Step 3: Word Embeddings

The cleaned text is converted into numerical vectors using trained tokenizer embeddings.

---

## Step 4: Deep Learning Prediction

The processed text passes through:

```text
CNN → BiLSTM → Capsule Layer → Softmax Output
```

---

## Step 5: Prediction Output

The system predicts the mental health category with probabilities.

---

## Step 6: LIME Explainability

LIME identifies important words responsible for the prediction.

---

# 🧩 Explanation of Important Files

## 📄 app.py

Main Streamlit application.

### Responsibilities

- Creates UI
- Takes user input
- Calls prediction functions
- Displays probabilities
- Shows LIME explanation

---

## 📄 utils/preprocess.py

Handles text preprocessing.

### Functions

- lowercase conversion
- regex cleaning
- stopword removal

---

## 📄 utils/predict.py

Core prediction pipeline.

### Responsibilities

- Loads trained model
- Loads tokenizer
- Loads encoder
- Predicts probabilities
- Generates LIME explanation

---

## 📄 utils/capsule_layer.py

Custom Capsule Network layer implementation.

### Responsibilities

- Capsule transformation
- vector representation
- routing logic
- custom layer loading support

---

## 📄 model/

Contains trained model artifacts.

| File | Purpose |
|------|----------|
| `.keras` | Trained deep learning model |
| `tokenizer.pkl` | Converts text → sequences |
| `encoder.pkl` | Converts labels |

---

# 📥 How to Clone the Repository

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Peesala-Naveen/Mental-Health-Classifier-Based-on-Social-Media-Text.git
```

---

## 2️⃣ Enter Project Folder

```bash
cd Mental-Health-Classifier-Based-on-Social-Media-Text
```

---

# 🛠️ Create Virtual Environment

## Windows

```bash
python -m venv venv
```

---

# ▶️ Activate Virtual Environment

```bash
venv\Scripts\activate
```

After activation:

```bash
(venv)
```

should appear in terminal.

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## IMPORTANT

Run Streamlit using:

```bash
python -m streamlit run app.py
```

⚠️ Do NOT use:

```bash
streamlit run app.py
```

because it may use global Python instead of the virtual environment.

---

# 🌐 Open the Application

After running:

```text
http://localhost:8501
```

will open automatically.

---

# 🧪 Example Input

```text
I feel tired, hopeless, and mentally exhausted every day.
```

---

# 📊 Example Output

```text
Prediction: Depression

Probabilities:
Depression: 0.91
Anxiety: 0.05
BPD: 0.02
```

LIME explanation will also display important words.

---

# 🧠 Deep Learning Architecture

## Proposed Architecture

```text
Input Text
   ↓
Preprocessing
   ↓
Tokenizer
   ↓
Padding
   ↓
Embedding Layer
   ↓
CNN Layer
   ↓
BiLSTM Layer
   ↓
Capsule Layer
   ↓
Softmax Output
```

---

# 📈 Mental Health Classes

The model predicts:

| Class |
|------|
| Depression |
| Anxiety |
| Bipolar Disorder |
| Borderline Personality Disorder |
| Schizophrenia |
| Autism |

---

# 📚 Dataset Used

## Dataset Source

- Reddit Mental Health Dataset

### Dataset Includes

- Depression posts
- Anxiety posts
- Bipolar disorder posts
- BPD posts
- Schizophrenia posts
- Autism posts

---

# 🔬 Machine Learning Techniques Used

## Traditional Models

- Naive Bayes
- XGBoost

## Deep Learning Models

- CNN
- BiGRU
- BiLSTM
- Conv-BiLSTM
- BiLSTM-Capsule
- Conv-BiLSTM-Capsule

---

# 🧠 Explainability Using LIME

LIME (Local Interpretable Model-Agnostic Explanations) explains:

- which words influenced prediction
- positive contributing words
- negative contributing words

This improves:

- transparency
- trustworthiness
- clinical interpretability

---

# 📌 Important Notes

## NLTK Stopwords

If stopwords are not downloaded automatically:

```bash
python
```

Then:

```python
import nltk
nltk.download('stopwords')
```

---

# ❗ Common Errors and Solutions

## 1️⃣ No module named 'lime'

### Solution

```bash
pip install lime
```

Then run:

```bash
python -m streamlit run app.py
```

---

## 2️⃣ TensorFlow Version Issues

### Solution

```bash
pip install tensorflow-cpu==2.16.1
```

---

## 3️⃣ Model Loading Error

Ensure these files exist inside `model/`:

```text
CONV_BILSTM_CAPS_Dynamic_Model.keras
CONV_BILSTM_CAPS_Dynamic_tokenizer.pkl
CONV_BILSTM_CAPS_Dynamic_encoder.pkl
```

---

## 4️⃣ Streamlit Not Opening

Run:

```bash
python -m streamlit run app.py
```

---

# 📋 Requirements

Main libraries:

```text
streamlit
tensorflow
numpy
scikit-learn
lime
nltk
matplotlib
```

---

# 🔥 Future Improvements

- Transformer Models (BERT, RoBERTa)
- SHAP Explainability
- Real-time API deployment
- Clinical dashboard integration
- Multi-modal mental health detection

---

# 👨‍💻 Authors

## Peesala Naveen

- B.Tech CSE (AI & DS)
- SASTRA Deemed University

---

## Kommineni Lokesh Naidu

- B.Tech CSE (AI & DS)
- SASTRA Deemed University

---

## Bathula Bhanu Prakash

- B.Tech CSE (AI & DS)
- SASTRA Deemed University

---

# 📜 License

This project is developed for educational and academic purposes.

---

# ⭐ GitHub Repository

```text
https://github.com/Peesala-Naveen/Mental-Health-Classifier-Based-on-Social-Media-Text
```

---

# ✅ Quick Start Commands

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Peesala-Naveen/Mental-Health-Classifier-Based-on-Social-Media-Text.git
```

---

## 2️⃣ Enter Folder

```bash
cd Mental-Health-Classifier-Based-on-Social-Media-Text
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 4️⃣ Activate Environment

```bash
venv\Scripts\activate
```

---

## 5️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Run Application

```bash
python -m streamlit run app.py
```

---

## 7️⃣ Open Browser

```text
http://localhost:8501
```

---

# 🎯 Project Status

- ✅ Completed
- ✅ Working
- ✅ GitHub Uploaded
- ✅ Streamlit Integrated
- ✅ LIME Explainability Added
