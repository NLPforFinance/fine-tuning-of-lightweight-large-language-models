from config import Config
from model_loader import SFTModelLoader, BaseModelLoader
from prompt_formatter import PromptFormatterShots, PromptFormatterSFT
from parse_label import LabelParser
from predictor import Predictor
from evaluate import BaseEvaluator, SFTEvaluator

class SentimentInferencePipeline:
    def __init__(self, model_parent_dir, output_parent_dir, test_data_path, model_type, mode = "shots", num_shots = 0):
        self.config = Config(model_parent_dir, output_parent_dir, test_data_path, model_type)
        if mode == "shots":
            self.model_loader = BaseModelLoader(self.config)
            self.prompt_formatter = PromptFormatterShots(self.config, num_shots, test_data_path) # for shot-learning, args as config, num_shots, test_data_path
            self.label_parser = LabelParser(model_type)
            self.predictor = Predictor(self.config, self.model_loader, self.prompt_formatter, self.label_parser)
            self.evaluator = BaseEvaluator(
                config=self.config,
                predictor=self.predictor,
                model_loader=self.model_loader,
                label_parser=self.label_parser
            )
        elif mode == "sft":
            self.model_loader = SFTModelLoader(self.config)
            self.prompt_formatter = PromptFormatterSFT(model_type) # for inference of SFT model and no shots (no examples in prompt)
            self.label_parser = LabelParser(model_type)
            self.predictor = Predictor(self.config, self.model_loader, self.prompt_formatter, self.label_parser)
            self.evaluator = SFTEvaluator(
                config=self.config,
                predictor=self.predictor,
                model_loader=self.model_loader,
                label_parser=self.label_parser
            )

    
    def run_all_splits(self):
        self.evaluator.run_all_splits()