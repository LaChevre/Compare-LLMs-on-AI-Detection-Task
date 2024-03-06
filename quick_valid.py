from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model
model_checkpoint = "C:/Users/Alex/Desktop/code/detect_ia/model/bert-trained/augmented_data_W/training_output/checkpoint-20514"
model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Assuming the device is set to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load validation data
#valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/out_validation_all_zero.csv'
#valid_essays = pd.read_csv(valid_path)
valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/test_essays.parquet'
valid_essays = pd.read_parquet(valid_path)
print(valid_essays['generated'].value_counts())

# Tokenize the validation texts
tokenized_valid_texts = tokenizer(list(valid_essays['text']),
                                  return_tensors='pt',
                                  padding='max_length',
                                  truncation=True,
                                  max_length=256)

# Valid labels
valid_labels = valid_essays['generated'].tolist()

# Validation dataset class
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

valid_dataset = ValidDataset({'input_ids': tokenized_valid_texts['input_ids'], 'attention_mask': tokenized_valid_texts['attention_mask']}, valid_labels)

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='/tmp/trainer',
        per_device_eval_batch_size=8  # Set lower if you have CUDA memory issues
    ),
    compute_metrics=compute_metrics,
)

# Evaluate the model
results = trainer.evaluate(valid_dataset)
print(results)
