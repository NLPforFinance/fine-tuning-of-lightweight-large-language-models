from transformers import AutoModelForCausalLM, TrainingArguments
from config import Config
from data_loader import create_balanced_dataset, tokenize_dataset
from trainer import DomainBalancedTrainer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
import os

def setup_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM,
        num_train_epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        lr_scheduler_type=Config.LR_SCHEDULER,
        warmup_ratio=Config.WARMUP_RATIO,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        bf16=Config.BF16,
        fp16=not Config.BF16,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        disable_tqdm=False,
        eval_strategy="no",
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        save_total_limit=1,
        report_to="none"
    )

def train_on_proportion(proportion, tokenizer, bnb_config):
    print(f"\n{'='*60}")
    print(f"STARTING BALANCED TRAINING FOR PROPORTION: {proportion}%")
    print(f"{'='*60}")
    
    balanced_dataset = create_balanced_dataset(proportion)
    tokenized_dataset = tokenize_dataset(balanced_dataset, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if Config.BF16 else torch.float16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    domain_weights = {
        'FinancialPhraseBank-v1.0': 1.25,  # real data: 4000 -> 5000
        'fiqa-sentiment-data-with-scores': 4.17,  # 1200 -> 5000
        'gold-sentiment': 0.5,  # 10000 -> 5000
        'twitter-sentiment-data': 0.42,  # 12000 -> 5000
        'chinese-finance-data': 0.37 # 13700 -> 5000
    }
    
    training_args = setup_training_args(
        os.path.join(Config.OUTPUT_ROOT, f"qwen-{proportion}")
    )
    
    trainer = DomainBalancedTrainer(
        domain_weights=domain_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer._cache_domains(balanced_dataset)
    trainer.train()
    
    model.save_pretrained(os.path.join(Config.OUTPUT_ROOT, f"qwen-{proportion}"))
    tokenizer.save_pretrained(os.path.join(Config.OUTPUT_ROOT, f"qwen-{proportion}"))
    print(f"\nTraining complete for proportion {proportion}%")