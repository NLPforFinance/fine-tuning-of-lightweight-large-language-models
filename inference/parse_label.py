import re

class LabelParser:
    def __init__(self, model_type):
        self.model_type = model_type.lower()
        self.label_pattern = re.compile(r'(positive|negative|neutral)', re.IGNORECASE)

    def extract_label(self, text):
        match = self.label_pattern.search(text.lower())
        return match.group(1).lower() if match else None

    def parse_output(self, full_text, prompt):
        if full_text.startswith(prompt):
            trimmed = full_text[len(prompt):].strip()
        else:
            trimmed = full_text.split("Output:")[-1].strip()

        match = re.search(r"\b(positive|negative|neutral)\b", trimmed, re.IGNORECASE)
        return match.group(1).lower() if match else None

    def parse_qwen(self, full_text, prompt):
        output_part = full_text.split("/no_think")[-1].strip()
        if "</think>" in output_part:
            output_part = output_part.split("</think>", 1)[-1].strip()

        match = re.search(r"\b(positive|negative|neutral)\b", output_part, re.IGNORECASE)
        return match.group(1).lower() if match else None

    def parse_ua_qwen(self, full_text, prompt):
        split_marker = "assistant\n"
        if split_marker in full_text:
            output_part = full_text.split(split_marker, 1)[-1].strip()
        else:
            output_part = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

        match = re.search(r"\b(positive|negative|neutral)\b", output_part, re.IGNORECASE)
        return match.group(1).lower() if match else None

    def parse_deepseek_ua(self, full_text, prompt=None):
        print("text:",full_text)
        assistant_blocks = re.split(r"<\|assistant\|>\n", full_text)
        for block in assistant_blocks[1:]:  
            if "<|user|>" in block:
                block = block.split("<|user|>", 1)[0]

            match = re.search(r"\b(positive|negative|neutral)\b", block, re.IGNORECASE)
            if match:
                label = match.group(1).lower()
                print("match: ",label)
                return label

        return None
    
    def parse_deepseek_ua_fewshots(self, full_text, prompt=None):
        output = full_text[len(prompt):]


        match = re.search(r"\b(positive|negative|neutral)\b", output, re.IGNORECASE)
        if match:
            label = match.group(1).lower()
            return label

        return None

    def parse_llama(self, full_text, prompt):
        assistant_split = full_text.split("assistant", 1)
        if len(assistant_split) > 1:
            trimmed = assistant_split[1].strip()
        else:
            trimmed = full_text.strip()

        match = re.search(r"\b(positive|negative|neutral)", trimmed, re.IGNORECASE)
        label = match.group(1).lower() if match else None
        return label
