import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# --- Load Model and Tokenizer from Hugging Face ---
MODEL_NAME = "Swapnil357/hate-speech-detector-bert" 
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# UI
st.title("🛡️ Hate Speech Detection using BERT")
st.write("Enter a sentence and get predictions for hate speech, offensive language, or neither.")

text_input = st.text_area("Enter your sentence here:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize and predict
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        # Labels
        labels = ["Hate Speech", "Offensive", "Neither"]
        result = {labels[i]: float(probs[i]) * 100 for i in range(3)}

        st.subheader("Prediction Probabilities:")
        for label, prob in result.items():
            st.write(f"**{label}**: {prob:.2f}%")

        # Show highest
        prediction = labels[probs.argmax()]
        st.success(f"🧠 **Predicted Category:** {prediction}")
