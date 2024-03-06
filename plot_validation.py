import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Load the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_checkpoint = "C:/Users/Alex/Desktop/code/detect_ia/model/bert/winningdat/training_output/checkpoint-51805"
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load validation data
#valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/out_validation_missleading.csv'
#valid_essays = pd.read_csv(valid_path)
valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/valid_essays.parquet'
valid_essays = pd.read_parquet(valid_path)

# Define the dataset class
class ValidDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define the compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# Columns to evaluate
columns = ["text", "cyrillic_text", "text_with_typos", "text_with_extra_spaces_and_newlines"]

# Dictionary to store accuracies
accuracies = {}

# Loop through each column and evaluate
for column in columns:
    print(f"Evaluating on: {column}")
    tokenized_texts = tokenizer(list(valid_essays[column]),
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=256)
    valid_labels = valid_essays['generated'].tolist()
    valid_dataset = ValidDataset({'input_ids': tokenized_texts['input_ids'], 'attention_mask': tokenized_texts['attention_mask']}, valid_labels)
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='/tmp/trainer',
            per_device_eval_batch_size=8,
            # Add other necessary arguments
        ),
        compute_metrics=compute_metrics,  # Make sure to include this
    )

    
    # Evaluate the model
    # Evaluate the model
    results = trainer.evaluate(valid_dataset)
    print(results)
    accuracies[column] = results['eval_accuracy']  # Assuming you're calculating accuracy in compute_metrics


# Plotting accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.xlabel('Text Column')
plt.ylabel('Accuracy')
plt.title('Model Accuracy on Different Text Columns')
plt.xticks(rotation=45)
plt.show()
