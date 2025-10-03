import json
import random

class PromptFormatterSFT:
    def __init__(self, model_type):
        self.model_type = model_type.lower()
    
    def format_prompt(self, instruction, input_text):
        if self.model_type == "llama":
            START = "<|start_header_id|>"
            END = "<|end_header_id|>"
            return f"{START}user{END}\n{instruction}\n{input_text}\n{START}assistant{END}\n"

        if self.model_type == "qwen":
            return f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n/no_think"

        if self.model_type == "deepseek":
            return f"<|user|>\n{instruction}\n{input_text}\n<|assistant|>\n"
        
class PromptFormatterShots:
    def __init__(self, model_config, num_shots, test_data_path=None):
        self.config = model_config
        self.model_type = self.config.model_type
        self.num_shots = num_shots
        self.test_data_path = test_data_path
        self.examples = []
        
        if test_data_path:
            self.load_test_data(test_data_path)
        
        print(f"Building prompts with {num_shots}-shot learning")
    
    def load_test_data(self, test_data_path):
        """Load test data from JSON file"""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    self.examples = data
                else:
                    print("Warning: Test data should be a list")
        except FileNotFoundError:
            print(f"Warning: Test data file '{test_data_path}' not found")
        except json.JSONDecodeError:
            print("Warning: Invalid JSON format in test data file")
    
    def get_random_examples(self, num_examples=3):
        """Get random examples from test data"""
        if not self.examples:
            return []
        
        if len(self.examples) < num_examples:
            num_examples = len(self.examples)
        
        return random.sample(self.examples, num_examples)
    
    def format_examples_section(self, examples):
        """Format the examples section for the prompt"""
        if not examples:
            return ""
        
        examples_text = "Here are some examples:\n"
        for i, example in enumerate(examples, 1):
            example_str = f"{example['input']}: {example['output']}"
            examples_text += f"{example_str}\n"
        
        return examples_text
    
    def format_prompt(self, instruction, input_text, include_examples=True):
        if include_examples and self.examples:
            random_examples = self.get_random_examples(self.num_shots)
            examples_section = self.format_examples_section(random_examples)
            instruction = f"{instruction}\n{examples_section}"
        
        if self.model_type == "llama":
            START = "<|start_header_id|>"
            END = "<|end_header_id|>"
            return f"{START}user{END}\n{instruction}\n{input_text}\n{START}assistant{END}\n"

        elif self.model_type == "qwen":
            return f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n/no_think"

        elif self.model_type == "deepseek":
            return f"<|user|>\n{instruction}\nSentence to analyze:\n{input_text}\n<|assistant|>\n"
        