from datasets import load_dataset
from transformers import MarianTokenizer

# def prepare_data(example, tokenizer, max_length=128):
#     # Tokenize and pad the input and output sentences
#     inputs = tokenizer(example["en"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
#     outputs = tokenizer(example["de"], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

#     # Ensure labels are provided for language modeling tasks
#     inputs["labels"] = outputs["input_ids"].clone()

#     return inputs

def prepare_data(examples):
    en_data = list(map(lambda x:x['en'], examples))
    de_data = list(map(lambda x:x['de'], examples))    
    return en_data, de_data

def load_dataset_from_hf(dataset_name):
    return load_dataset(dataset_name)

# Save this file as utils.py
