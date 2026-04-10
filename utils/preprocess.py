import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Ensure nltk path
nltk.data.path.append("C:\\Users\\siddh\\nltk_data")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)