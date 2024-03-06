import os
import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import pandas as pd


def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))
        return (loss, outputs) if return_outputs else loss



# Set random seed for reproducibility
random.seed(0)
torch.manual_seed(0)

# Load the data from parquet files
train_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/train_essays.parquet'
valid_path = 'C:/Users/Alex/Desktop/code/detect_ia/data/better_data/valid_essays.parquet'
train_essays = pd.read_parquet(train_path)
valid_essays = pd.read_parquet(valid_path)

# Convert the dataframes to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_essays)
valid_dataset = Dataset.from_pandas(valid_essays)

# Combine into a DatasetDict
ds = DatasetDict({
    'train': train_dataset,
    'val': valid_dataset
})

# Model name (Update if using a different model)
model_name = 'microsoft/phi-2'

# Ensure the tokenizer uses the correct padding token
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)

# qunatization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id


training_args = TrainingArguments(
    output_dir = 'C:/Users/Alex/Desktop/code/detect_ia/model/phi-2_trained_LoRa/data_W/training_output',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8, # tested with 16gb gpu ram
    per_device_eval_batch_size = 8,
    num_train_epochs = 5,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True
)

# train
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_ds['train'],
    eval_dataset = tokenized_ds['val'],
    tokenizer = tokenizer,
    data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics = compute_metrics,
)

trainer.train()

# save model
peft_model_id = 'C:/Users/Alex/Desktop/code/detect_ia/model/phi-2_peft_model/data_W/'
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)