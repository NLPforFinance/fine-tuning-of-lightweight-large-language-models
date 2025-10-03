import json
import os
from datasets import Dataset, concatenate_datasets
from config import Config

def load_and_format_data(file_path, domain_name):
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        formatted = []
        for item in data:
            if not all(key in item for key in ['instruction', 'input', 'output']):
                continue
                
            formatted.append({
                "instruction": str(item['instruction']),
                "input": str(item['input']),
                "output": str(item['output']),
                "domain": domain_name
            })
        return formatted[:Config.MAX_SAMPLES]
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        raise

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        texts = []
        prompts = []
        domains = []
        
        for i in range(len(examples['instruction'])):
            START = "<|start_header_id|>"
            END = "<|end_header_id|>"

            prompt = f"{START}user{END}\n{examples['instruction'][i]}\n{examples['input'][i]}"
            full_text = f"{prompt}\n{START}assistant{END}\n{examples['output'][i]}{END}"
            
            if len(tokenizer.encode(prompt)) < Config.MIN_LENGTH:
                continue
                
            texts.append(full_text)
            prompts.append(prompt)
            domains.append(examples['domain'][i])
        
        if not texts:
            return {}
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        prompt_tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        for i in range(len(prompts)):
            prompt_len = (prompt_tokenized["input_ids"][i] != tokenizer.pad_token_id).sum().item()
            labels[i, :prompt_len] = -100
            
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
    
    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=8,
        remove_columns=["instruction", "input", "output", "domain"] 
    ).filter(lambda x: len(x["input_ids"]) > 0)

def create_balanced_dataset(proportion):
    all_datasets = []
    domain_counts = {}
    
    for dataset_path in Config.DATASET_PARENT_PATHS:
        domain_name = os.path.basename(os.path.dirname(dataset_path))
        data_file = os.path.join(dataset_path, f"train_data_{proportion}.json")
        
        try:
            data = load_and_format_data(data_file, domain_name)
            dataset = Dataset.from_list(data)
            
            target_size = int(Config.TARGET_SIZES[domain_name] * (proportion / 100))
            if len(dataset) > target_size:
                dataset = dataset.select(range(target_size))
            else:
                repeat_factor = (target_size // len(dataset)) + 1
                dataset = concatenate_datasets([dataset] * repeat_factor)
                dataset = dataset.select(range(target_size))
            
            domain_counts[domain_name] = len(dataset)
            all_datasets.append(dataset)
            
        except FileNotFoundError:
            print(f"Warning: Skipping missing file {data_file}")
            continue
    
    if not all_datasets:
        raise ValueError("No valid datasets found for training")
    
    print(f"\nDataset sizes for proportion {proportion}%:")
    for domain, count in domain_counts.items():
        print(f"{domain}: {count} samples")
    
    return concatenate_datasets(all_datasets).shuffle(seed=42)