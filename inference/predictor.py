import torch 

class Predictor:
    def __init__(self, config, model_loader, prompt_formatter, label_parser):
        self.config = config
        self.model_loader = model_loader
        self.prompt_formatter = prompt_formatter
        self.label_parser = label_parser

    def predict(self, model, tokenizer, instruction, input_text, max_new_tokens=10):
        prompt = self.prompt_formatter.format_prompt(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.label_parser.extract_label(output_text.split("Output:")[-1])

    def predict_batch(self, model, tokenizer, instructions, input_texts, max_new_tokens=100):
        prompts = [
            self.prompt_formatter.format_prompt(instr, text)
            for instr, text in zip(instructions, input_texts)
        ]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=256,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if self.config.model_type == "qwen":
            return [self.label_parser.parse_qwen(full_text, prompt) 
                   for prompt, full_text in zip(prompts, decoded)]
        elif self.config.model_type == "deepseek":
            return [self.label_parser.parse_deepseek_ua_fewshots(full_text, prompt) 
                   for prompt, full_text in zip(prompts, decoded)]
        elif self.config.model_type == "llama":
            return [self.label_parser.parse_llama(full_text, prompt) 
                   for prompt, full_text in zip(prompts, decoded)]
        else:
            return [self.label_parser.extract_label(text.split("Output:")[-1]) 
                   for text in decoded]