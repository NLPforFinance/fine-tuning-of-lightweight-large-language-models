import warnings
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from config import Config
from training_utils import train_on_proportion

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    for prop in Config.PROPORTIONS:
        train_on_proportion(prop, tokenizer, bnb_config)
