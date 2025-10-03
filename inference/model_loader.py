from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import torch

class SFTModelLoader:
    def __init__(self, config):
        self.config = config
    
    def load_model(self, model_path):
        base_model_name = self.config.base_models[self.config.model_type]
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        
        is_merged_model = not any(
            file.endswith("adapter_config.json") 
            for file in os.listdir(model_path)
        )
        if is_merged_model:
            print(f"Loading merged model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=self.config.bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            print(f"Loading PEFT adapter from {model_path}")
            print(f"Base model name: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=self.config.bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            model = PeftModel.from_pretrained(model, model_path)
        
        model.eval()
        return model, tokenizer
    
class BaseModelLoader:
    def __init__(self, config):
        self.config = config
    
    def load_model(self, model_path=None):
        base_model_name = self.config.base_models[self.config.model_type]
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=self.config.bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        model.eval()
        return model, tokenizer