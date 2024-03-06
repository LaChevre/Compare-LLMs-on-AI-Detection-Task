from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import set_seed
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

set_seed(42)  # Set a random seed for reproducibility

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

# Specify the model name from Hugging Face
huggingface_model_name = 'microsoft/phi-2-mini'

# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

# Set the tokenizer's padding token to its EOS token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(huggingface_model_name, num_labels=2)

# Move the model to GPU if CUDA is available
if cuda_available:
    model.cuda()

# Load the data
train_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/train_essays.parquet'
valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/valid_essays.parquet'
train_essays = pd.read_parquet(train_path)
valid_essays = pd.read_parquet(valid_path)

print(train_essays.head())

# Tokenize the text
max_length = 256
tokenized_train_texts = tokenizer(list(train_essays['text']), return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
tokenized_valid_texts = tokenizer(list(valid_essays['text']), return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

# Dataset class
class EssayDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Prepare dataset
train_labels = train_essays['generated'].tolist()
valid_labels = valid_essays['generated'].tolist()
train_dataset = EssayDataset(tokenized_train_texts, train_labels)
valid_dataset = EssayDataset(tokenized_valid_texts, valid_labels)

# Training arguments
# Adjusted training arguments with reduced batch size and enabled mixed precision
training_args = TrainingArguments(
    output_dir='C:/Users/Alex/Desktop/code/detect_ia/model/phi-2_trained/data_W/training_output',
    num_train_epochs=5,
    per_device_train_batch_size=4,  # Reduced batch size
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_total_limit=2,
    logging_dir='C:/Users/Alex/Desktop/code/detect_ia/model/phi-2_trained/logs',
    logging_steps=10,
    fp16=True,  # Enable mixed precision
)

# Your existing code for model, tokenizer, dataset preparation remains the same

# Trainer initialization with adjusted training arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Start training
print("Starting training...")
trainer.train()
print("Training completed!")
