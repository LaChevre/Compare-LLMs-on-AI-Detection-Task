import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Training will run on CPU.")

torch.cuda.set_per_process_memory_fraction(0.85, 0)
torch.backends.cuda.matmul.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'



def train_bert(train_path, valid_path, save_path):
    # Load the data
    train_essays = pd.read_parquet(train_path)
    #train_essays = pd.read_csv(train_path)
    valid_essays = pd.read_parquet(valid_path)
    #valid_essays = pd.read_csv(valid_path)
    print(train_essays.head())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Starting tokenization...")
    max_length = 256
    tokenized_train_texts = tokenizer(list(train_essays['text']),
                                    return_tensors='pt',
                                    padding='max_length', 
                                    truncation=True,
                                    max_length=max_length)

    tokenized_valid_texts = tokenizer(list(valid_essays['text']),
                                    return_tensors='pt',
                                    padding='max_length', 
                                    truncation=True,
                                    max_length=max_length)
    print("Tokenization complete!")

    class EssayDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_labels = train_essays['generated'].tolist()
    valid_labels = valid_essays['generated'].tolist()

    train_dataset = EssayDataset(tokenized_train_texts, train_labels)
    valid_dataset = EssayDataset(tokenized_valid_texts, valid_labels)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    if cuda_available:
        model = model.to('cuda')

    training_args = TrainingArguments(
        output_dir=save_path+'training_output',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_dir=save_path+'logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed!")

#train_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/train_essays.parquet'
#valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/valid_essays.parquet'
#train_bert(train_path, valid_path, 'C:/Users/Alex/Desktop/code/detect_ia/model/bert-trained/data_W/')

train_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/augmented_train_essays.parquet'
valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/augmented_valid_essays.parquet'
train_bert(train_path, valid_path, 'C:/Users/Alex/Desktop/code/detect_ia/model/bert-trained/augmented_data_W/')   