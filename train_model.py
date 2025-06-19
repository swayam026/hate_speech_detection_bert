import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load dataset
df = pd.read_csv("labeled_data.csv")
df = df[['tweet', 'class']].dropna()
df.columns = ['text', 'label']

# Map class: 0 = hate speech, 1 = offensive, 2 = neither
label_map = {0: 0, 1: 1, 2: 2}
df['label'] = df['label'].map(label_map)

# Split data
train_texts, test_texts = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HuggingFace Dataset
train_ds = Dataset.from_pandas(train_texts)
test_ds = Dataset.from_pandas(test_texts)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function with padding & truncation
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments (minimal, compatible version)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    do_train=True,
    do_eval=True,
    logging_dir='./logs',
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

# Start training
trainer.train()

# Save model and tokenizer
model.save_pretrained("hate_speech_bert_model")
tokenizer.save_pretrained("hate_speech_bert_model")
