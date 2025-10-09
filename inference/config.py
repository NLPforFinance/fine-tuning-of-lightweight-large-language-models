import torch
from pathlib import Path
from transformers import BitsAndBytesConfig
import re

class Config:
    def __init__(self, model_parent_dir, output_parent_dir, test_data_path, model_type):
        self.model_type = model_type.lower()
        self.model_parent_dir = Path(model_parent_dir)
        self.output_parent_dir = Path(output_parent_dir)
        self.test_data_path = Path(test_data_path)
        self.output_parent_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.base_models = {
            "deepseek": "deepseek-ai/deepseek-llm-7b-base",
            "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
            "qwen": "Qwen/Qwen3-8B"
        }
        
        self.label_pattern = re.compile(r'(positive|negative|neutral)', re.IGNORECASE)
